"""Per-image non-regression gate. A pass is research evidence, never deployment approval."""
import math


def paired_quality_gate(baseline, candidate):
    def index(rows):
        mapped = {}
        for row in rows:
            key = row['imageId']
            if key in mapped: raise ValueError('Duplicate image in quality evidence')
            metrics = row['metrics']
            for name in ('hand_exact_match', 'model_exact', 'accepted', 'failed'):
                if type(metrics[name]) is not bool: raise ValueError('Boolean quality flags required')
            mapped[key] = metrics
        return mapped
    old, new = index(baseline), index(candidate)
    if not old or old.keys() != new.keys(): raise ValueError('Compare the same complete image IDs')
    if any(old[i]['truth_count'] != new[i]['truth_count'] for i in old):
        raise ValueError('Ground-truth count changed')
    lost_auto = sorted(i for i in old if old[i]['hand_exact_match'] and not new[i]['hand_exact_match'])
    lost_model = sorted(i for i in old if old[i]['model_exact'] and not new[i]['model_exact'])
    wrong = lambda m: m['accepted'] and not m['model_exact']
    new_wrong = sorted(i for i in old if wrong(new[i]) and not wrong(old[i]))
    failed = sorted(i for i in new if new[i]['failed'])
    # A broken reference cannot justify a candidate pass.
    baseline_failed = sorted(i for i in old if old[i]['failed'])
    return {'passed': not (lost_auto or lost_model or new_wrong or failed or baseline_failed),
            'images': len(old), 'lostAutomaticExactIds': lost_auto, 'lostModelExactIds': lost_model,
            'newWrongAcceptIds': new_wrong, 'candidateFailedIds': failed, 'baselineFailedIds': baseline_failed,
            'automaticExactGains': sorted(i for i in old if not old[i]['hand_exact_match'] and new[i]['hand_exact_match']),
            'automaticPromotion': False}


def percentile(values, q):
    if not values or not 0 <= q <= 1 or not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError('Finite nonnegative timing samples required')
    values = sorted(values); position = (len(values)-1)*q
    low, high = math.floor(position), math.ceil(position)
    return values[low]+(values[high]-values[low])*(position-low)


def stages(truth, predictions, threshold=.5):
    """Class-blind maximum-cardinality IoU matching, then classify those same pairs."""
    from ml.eval.metrics import iou
    edges = [sorted((j for j,p in enumerate(predictions) if iou(t,p)>=threshold),
                    key=lambda j: (-iou(t,predictions[j]), j)) for t in truth]
    assigned = {}
    def augment(t, seen):
        for p in edges[t]:
            if p in seen: continue
            seen.add(p)
            if p not in assigned or augment(assigned[p], seen): assigned[p]=t; return True
        return False
    for t in range(len(truth)): augment(t,set())
    return {'truth':len(truth),'proposals':len(predictions),'localized':len(assigned),
            'correctClassesOnLocalizedPairs':sum(truth[t].class_id==predictions[p].class_id for p,t in assigned.items())}
