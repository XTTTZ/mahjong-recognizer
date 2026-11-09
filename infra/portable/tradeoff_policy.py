"""Provisional measured tradeoffs after explicit acceptance of small accuracy loss.

This aggregate policy intentionally differs from the historical per-image
non-regression gate. It supports controlled evaluation/use; it never deploys a
model or certifies accuracy on new sources, phone hardware, or a sealed test.
"""
from __future__ import annotations
import copy
import math
from infra.portable.quality_gate import paired_quality_gate

ANNOTATION_SHA = '42758d7a3a7bec0a4f6fd415c0de5b5a76482a33de1ce45d91c37df4be1abc30'
COHORTS = ('all150', 'primary82', 'thirteenFourteen45')
DEFAULT_POLICY = {
    'id': 'portable-v10-provisional-tradeoff-20261109',
    'expectedImages': {'all150': 150, 'primary82': 82, 'thirteenFourteen45': 45},
    'maxAutomaticExactDrop': {'all150': 3, 'primary82': 2, 'thirteenFourteen45': 1},
    'maxWrongAcceptIncrease': {'all150': 1, 'primary82': 1, 'thirteenFourteen45': 1},
    'maximumFailedRequests': 0,
    'normalHandP95TargetMs': 2000.,
    'minimumPairedLatencyReduction': .20,
    'minimumModelBytesReduction': .30,
    'purpose': 'Provisional engineering choice following user-authorized small accuracy/speed tradeoff; not statistical proof',
}


def _count(value, maximum, label):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(label + ' must be an integer within its cohort denominator')
    return value


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(label + ' must be finite and nonnegative')
    return float(value)


def _policy(value):
    policy = copy.deepcopy(DEFAULT_POLICY if value is None else value)
    if set(policy['expectedImages']) != set(COHORTS) or policy['expectedImages'] != DEFAULT_POLICY['expectedImages']:
        raise ValueError('This policy only qualifies the complete frozen150/82/45 cohorts')
    for name in ('maxAutomaticExactDrop', 'maxWrongAcceptIncrease'):
        if set(policy[name]) != set(COHORTS): raise ValueError('Every cohort needs an explicit tolerance')
        for cohort, count in policy[name].items(): _count(count, policy['expectedImages'][cohort], name)
    _count(policy['maximumFailedRequests'], 0, 'maximumFailedRequests')
    for name in ('normalHandP95TargetMs', 'minimumPairedLatencyReduction', 'minimumModelBytesReduction'):
        _finite(policy[name], name)
    if not policy['normalHandP95TargetMs'] or any(not 0 < policy[n] < 1 for n in ('minimumPairedLatencyReduction', 'minimumModelBytesReduction')):
        raise ValueError('Use a positive runtime target and fractional meaningful-benefit thresholds')
    return policy


def _summary(summary, policy):
    if not isinstance(summary, dict) or not set(COHORTS) <= summary.keys():
        raise ValueError('All frozen cohorts must be reported for every model')
    for cohort in COHORTS:
        row = summary[cohort]; size = policy['expectedImages'][cohort]
        if row.get('images') != size or type(row.get('images')) is not int:
            raise ValueError('Partial or altered denominator: ' + cohort)
        for key in ('successes', 'wrongAccepts', 'failed_requests'):
            _count(row.get(key), size, cohort + '/' + key)
        if row['successes'] + row['wrongAccepts'] + row['failed_requests'] > size:
            raise ValueError('Correct accepts, wrong accepts and failures must be disjoint')
        if 'timedImages' in row: _count(row['timedImages'], size, cohort + '/timedImages')
        if row.get('p95Ms') is not None: _finite(row['p95Ms'], cohort + '/p95Ms')


def _rows_check(rows, summary, policy):
    if len(rows) != 150 or len({r['imageId'] for r in rows}) != 150:
        raise ValueError('Per-image evidence must contain150 unique image IDs')
    for row in rows:
        m = row['metrics']
        if type(m.get('truth_count')) is not int or m['truth_count'] < 1:
            raise ValueError('Positive integer ground-truth counts required')
        for key in ('hand_exact_match', 'model_exact', 'accepted', 'failed'):
            if type(m.get(key)) is not bool: raise ValueError('Boolean per-image flags required')
        if m['hand_exact_match'] != (m['model_exact'] and m['accepted'] and not m['failed']):
            raise ValueError('Automatic exact flag disagrees with model/acceptance/failure flags')
        if m['failed'] and (m['accepted'] or m['model_exact']): raise ValueError('A failed inference cannot be accepted or model-exact')
    scopes = {'all150': rows, 'primary82': [r for r in rows if r['metrics']['truth_count'] >= 10],
              'thirteenFourteen45': [r for r in rows if r['metrics']['truth_count'] in (13, 14)]}
    for cohort, selected in scopes.items():
        current = summary[cohort]
        actual = {'images': len(selected), 'successes': sum(r['metrics']['hand_exact_match'] for r in selected),
                  'wrongAccepts': sum(r['metrics']['accepted'] and not r['metrics']['model_exact'] for r in selected),
                  'failed_requests': sum(r['metrics']['failed'] for r in selected)}
        if any(current[k] != v for k, v in actual.items()):
            raise ValueError('Per-image flags disagree with aggregate evidence: ' + cohort)


def evaluate_matrix_tradeoffs(report, baseline_key='baseline', policy=None):
    """Assess a complete same-hardware matrix without changing thresholds or data.

    Required report keys: partial=False, annotationSha256, thresholds,
    summaries[model][cohort] (images/successes/wrongAccepts/failed_requests/p95Ms),
    and normal cohort timedImages for runtime qualification. modelBytes and
    pairedOnSameHardware supply evidence of a meaningful size/latency benefit.
    Optional perImage[model] preserves and validates historical identity checks.
    """
    policy = _policy(policy)
    if report.get('partial') is not False or report.get('annotationSha256') != ANNOTATION_SHA:
        raise ValueError('Require the complete frozen validation annotation, never partial/sealed-test evidence')
    if report.get('thresholds') != {'detector': .05, 'fused': .35, 'accept': .7}:
        raise ValueError('Tradeoffs must use the unchanged production operating point')
    models = report.get('summaries', {})
    if baseline_key not in models or len(models) < 2:
        raise ValueError('A baseline and at least one candidate are required')
    for name, summary in models.items():
        _summary(summary, policy)
        if name in report.get('perImage', {}): _rows_check(report['perImage'][name], summary, policy)
    baseline = models[baseline_key]
    if any(baseline[c]['failed_requests'] for c in COHORTS):
        raise ValueError('A failed baseline cannot qualify a candidate')
    sizes = report.get('modelBytes', {})
    for name, size in sizes.items():
        if name not in models or type(size) is not int or size <= 0:
            raise ValueError('Model sizes must be positive measured byte counts for matrix models')
    execution = report.get('execution')
    supported_context = execution in ('modal-cpu', 'local-cpu', 'browser-wasm', 'browser-webgpu')
    assessments = {}
    for name, summary in models.items():
        if name == baseline_key: continue
        comparisons, reasons = {}, []
        for cohort in COHORTS:
            old, new = baseline[cohort], summary[cohort]
            drop = max(0, old['successes'] - new['successes'])
            wrong_increase = new['wrongAccepts'] - old['wrongAccepts']
            comparisons[cohort] = {'images': new['images'], 'baselineAutomaticExact': old['successes'],
                'candidateAutomaticExact': new['successes'], 'automaticExactChange': new['successes']-old['successes'],
                'baselineWrongAccepts': old['wrongAccepts'], 'candidateWrongAccepts': new['wrongAccepts'],
                'wrongAcceptChange': wrong_increase, 'failedRequests': new['failed_requests']}
            if drop > policy['maxAutomaticExactDrop'][cohort]: reasons.append(cohort + ':automatic_exact_drop')
            if wrong_increase > policy['maxWrongAcceptIncrease'][cohort]: reasons.append(cohort + ':wrong_accept_increase')
            if new['failed_requests'] > policy['maximumFailedRequests']: reasons.append(cohort + ':inference_failures')
        old, new = baseline['thirteenFourteen45'], summary['thirteenFourteen45']
        complete_times = new.get('timedImages') == 45 and new.get('p95Ms') is not None
        latency_pass = supported_context and complete_times and new['p95Ms'] <= policy['normalHandP95TargetMs']
        paired = report.get('pairedOnSameHardware') is True
        latency_gain = None
        if paired and complete_times and old.get('timedImages') == 45 and old.get('p95Ms', 0) and old['p95Ms'] > 0:
            latency_gain = 1 - new['p95Ms']/old['p95Ms']
        size_gain = 1 - sizes[name]/sizes[baseline_key] if name in sizes and baseline_key in sizes else None
        meaningful = ((latency_gain is not None and latency_gain >= policy['minimumPairedLatencyReduction']) or
                      (size_gain is not None and size_gain >= policy['minimumModelBytesReduction']))
        strict = None
        rows = report.get('perImage', {})
        if baseline_key in rows and name in rows:
            strict = paired_quality_gate(rows[baseline_key], rows[name])
        qualified = not reasons and latency_pass and meaningful
        assessments[name] = {'qualityPassed': not reasons, 'qualityFailureReasons': reasons,
            'cohorts': comparisons, 'historicalStrictGateDiagnostic': strict,
            'runtime': {'execution': execution, 'normalHandP95Ms': new.get('p95Ms'),
                        'normalHandTimedImages': new.get('timedImages'), 'targetPassed': latency_pass,
                        'completeTargetRuntimeEvidence': bool(supported_context and complete_times),
                        'pairedP95Reduction': latency_gain, 'modelBytesReduction': size_gain,
                        'meaningfulBenefit': bool(meaningful)},
            'qualifiedForControlledUse': bool(qualified),
            'decision': 'qualified_for_controlled_use' if qualified else 'quality_not_qualified' if reasons else
                        'runtime_or_benefit_not_established',
            'wrongAcceptIncreaseRequiresReview': any(c['wrongAcceptChange'] > 0 for c in comparisons.values()) or
                                                bool(strict and strict['newWrongAcceptIds']),
            'automaticPromotion': False}
    return {'policy': policy, 'baselineKey': baseline_key, 'candidateAssessments': assessments,
        'automaticPromotion': False, 'phoneHardwareTested': report.get('phoneHardwareTested') is True,
        'limitations': ['Small frozen validation set with related source groups; aggregate tolerance is an engineering choice, not proof of non-inferiority.',
                       'Per-image losses and newly wrong accepts remain visible even when aggregate quality qualifies.',
                       'Warm pipeline target excludes download/load/network/cold start; those need separate measurements.',
                       'Desktop browser results do not establish physical-phone speed or accuracy.',
                       'Qualification never changes the website or deploys a model.']}
