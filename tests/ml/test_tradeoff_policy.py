import copy
import math
import unittest
from infra.portable.tradeoff_policy import ANNOTATION_SHA,evaluate_matrix_tradeoffs


def fixture():
    def model(successes,wrong=(3,0,0),p95=1800):
        return {name:{'images':n,'successes':s,'wrongAccepts':w,'failed_requests':0,'p95Ms':p95,'timedImages':n}
                for name,n,s,w in zip(('all150','primary82','thirteenFourteen45'),(150,82,45),successes,wrong)}
    return {'partial':False,'annotationSha256':ANNOTATION_SHA,'thresholds':{'detector':.05,'fused':.35,'accept':.7},
            'execution':'modal-cpu','pairedOnSameHardware':True,'modelBytes':{'baseline':119000000,'compact':29000000},
            'summaries':{'baseline':model((135,72,43)), 'compact':model((132,70,42),p95=650)}}


def paired_rows():
    base, compact = [], []
    accepted_exact = set(range(43)) | set(range(45,74)) | set(range(82,145))
    for ident in range(150):
        count = 13 if ident < 45 else 10 if ident < 82 else 1
        exact = ident in accepted_exact or ident in (43,74)
        accepted = ident in accepted_exact or ident in (145,146,147)
        metric = {'truth_count':count,'model_exact':exact,'accepted':accepted,
                  'hand_exact_match':exact and accepted,'failed':False}
        base.append({'imageId':ident,'metrics':metric})
        changed = dict(metric)
        if ident in (0,45,82):
            changed.update(model_exact=False,accepted=False,hand_exact_match=False)
        compact.append({'imageId':ident,'metrics':changed})
    return {'baseline':base,'compact':compact}


class TradeoffPolicyTests(unittest.TestCase):
    def test_small_aggregate_losses_can_qualify_and_no_automatic_promotion(self):
        result=evaluate_matrix_tradeoffs(fixture());decision=result['candidateAssessments']['compact']
        self.assertTrue(decision['qualityPassed']);self.assertTrue(decision['qualifiedForControlledUse'])
        self.assertFalse(result['automaticPromotion']);self.assertFalse(decision['automaticPromotion'])
        self.assertAlmostEqual(decision['runtime']['modelBytesReduction'],1-29/119)
    def test_every_cohort_has_its_own_drop_and_wrong_accept_cap(self):
        for cohort,key,value,reason in [('all150','successes',131,'automatic_exact_drop'),
                                       ('primary82','successes',69,'automatic_exact_drop'),
                                       ('thirteenFourteen45','successes',41,'automatic_exact_drop'),
                                       ('primary82','wrongAccepts',2,'wrong_accept_increase')]:
            with self.subTest(cohort=cohort,key=key):
                f=fixture();f['summaries']['compact'][cohort][key]=value
                decision=evaluate_matrix_tradeoffs(f)['candidateAssessments']['compact']
                self.assertFalse(decision['qualifiedForControlledUse']);self.assertIn(cohort+':'+reason,decision['qualityFailureReasons'])
        f=fixture();f['summaries']['compact']['primary82']['wrongAccepts']=1
        decision=evaluate_matrix_tradeoffs(f)['candidateAssessments']['compact']
        self.assertTrue(decision['qualityPassed']);self.assertTrue(decision['wrongAcceptIncreaseRequiresReview'])
    def test_failures_cannot_be_hidden_by_good_accuracy_or_speed(self):
        f=fixture();f['summaries']['compact']['all150']['failed_requests']=1
        decision=evaluate_matrix_tradeoffs(f)['candidateAssessments']['compact']
        self.assertFalse(decision['qualityPassed']);self.assertIn('all150:inference_failures',decision['qualityFailureReasons'])
        f=fixture();f['summaries']['baseline']['all150']['failed_requests']=1
        with self.assertRaisesRegex(ValueError,'baseline'):evaluate_matrix_tradeoffs(f)
    def test_partial_changed_denominator_or_operating_point_rejected(self):
        f=fixture();f['partial']=True
        with self.assertRaises(ValueError):evaluate_matrix_tradeoffs(f)
        for key,value in [('images',44),('successes',45.0),('wrongAccepts',True)]:
            f=fixture();f['summaries']['compact']['thirteenFourteen45'][key]=value
            with self.assertRaises(ValueError):evaluate_matrix_tradeoffs(f)
        f=fixture();f['thresholds']['fused']=.4
        with self.assertRaisesRegex(ValueError,'operating point'):evaluate_matrix_tradeoffs(f)
    def test_runtime_needs_actual_target_context_complete_timings_and_meaningful_benefit(self):
        for change in ('missing_count','missing_p95','wrong_context','slow'):
            f=fixture();normal=f['summaries']['compact']['thirteenFourteen45']
            if change=='missing_count':del normal['timedImages']
            elif change=='missing_p95':normal['p95Ms']=None
            elif change=='wrong_context':f['execution']='desktop-layout-only'
            else:normal['p95Ms']=2001
            self.assertFalse(evaluate_matrix_tradeoffs(f)['candidateAssessments']['compact']['qualifiedForControlledUse'])
        f=fixture();f['modelBytes']['compact']=115000000
        f['summaries']['compact']['thirteenFourteen45']['p95Ms']=1700
        decision=evaluate_matrix_tradeoffs(f)['candidateAssessments']['compact']
        self.assertTrue(decision['runtime']['targetPassed']);self.assertFalse(decision['runtime']['meaningfulBenefit'])
        self.assertFalse(decision['qualifiedForControlledUse'])
    def test_invalid_finite_sizes_and_timings_and_failed_reference_rejected(self):
        for invalid in (float('nan'),float('inf'),-1,True):
            f=fixture();f['summaries']['compact']['thirteenFourteen45']['p95Ms']=invalid
            with self.assertRaises(ValueError):evaluate_matrix_tradeoffs(f)
        f=fixture();f['modelBytes']['compact']=True
        with self.assertRaises(ValueError):evaluate_matrix_tradeoffs(f)
    def test_claimed_phone_evidence_is_not_inferred_from_browser_context(self):
        f=fixture();f['execution']='browser-webgpu'
        r=evaluate_matrix_tradeoffs(f)
        self.assertFalse(r['phoneHardwareTested'])
        self.assertTrue(r['candidateAssessments']['compact']['qualifiedForControlledUse'])
    def test_historical_per_image_regressions_are_visible_but_no_longer_hard_block(self):
        f=fixture();f['perImage']=paired_rows()
        result=evaluate_matrix_tradeoffs(f)['candidateAssessments']['compact']
        self.assertFalse(result['historicalStrictGateDiagnostic']['passed'])
        self.assertEqual(result['historicalStrictGateDiagnostic']['lostAutomaticExactIds'],[0,45,82])
        self.assertTrue(result['qualityPassed']);self.assertTrue(result['qualifiedForControlledUse'])
    def test_inconsistent_or_incomplete_per_image_claims_cannot_borrow_good_summaries(self):
        for change in ('missing','fake_summary','contradictory_flags'):
            f=fixture();f['perImage']=paired_rows()
            if change=='missing':f['perImage']['compact'].pop()
            elif change=='fake_summary':f['summaries']['compact']['all150']['successes']=133
            else:f['perImage']['compact'][0]['metrics']['hand_exact_match']=True
            with self.assertRaises(ValueError):evaluate_matrix_tradeoffs(f)

if __name__=='__main__':unittest.main()
