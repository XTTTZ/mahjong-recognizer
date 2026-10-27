"""Strict image-level metrics. Failed/rejected requests remain in denominators.

Boxes use [x, y, width, height]. Matching is maximum bipartite matching, rather
than greedy IoU assignment (which can mis-score repeated, adjacent tiles).
"""
from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class Tile:
    class_id: str
    bbox: tuple[float, float, float, float]

    def __post_init__(self):
        if len(self.bbox) != 4 or not all(math.isfinite(v) for v in self.bbox):
            raise ValueError("bbox must contain four finite values")
        if self.bbox[2] <= 0 or self.bbox[3] <= 0:
            raise ValueError("bbox must have positive width and height")


def iou(a: Tile, b: Tile) -> float:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by))
    return overlap / (aw * ah + bw * bh - overlap)


def maximum_matches(truth: list[Tile], predictions: list[Tile], threshold=0.5, require_class=True) -> int:
    if not 0 <= threshold <= 1:
        raise ValueError("IoU threshold must be in [0,1]")
    edges = [[j for j, p in enumerate(predictions)
              if (not require_class or t.class_id == p.class_id) and iou(t, p) >= threshold]
             for t in truth]
    assigned: dict[int, int] = {}

    def augment(t: int, seen: set[int]) -> bool:
        for p in edges[t]:
            if p in seen:
                continue
            seen.add(p)
            if p not in assigned or augment(assigned[p], seen):
                assigned[p] = t
                return True
        return False

    return sum(augment(t, set()) for t in range(len(truth)))


def score_image(truth: Iterable[Tile], predictions: Iterable[Tile], *, accepted=True, failed=False, threshold=0.5) -> dict:
    truth, predictions = list(truth), list(predictions)
    matched = maximum_matches(truth, predictions, threshold)
    localized = maximum_matches(truth, predictions, threshold, require_class=False)
    model_exact = matched == len(truth) == len(predictions) and not failed
    return {
        "truth_count": len(truth), "prediction_count": len(predictions),
        "matched": matched, "missed_or_wrong": len(truth) - matched,
        "extra_or_wrong": len(predictions) - matched,
        "localization_matches": localized,
        "model_exact": model_exact,
        "hand_exact_match": model_exact and accepted and not failed,
        "multiset_exact": not failed and Counter(t.class_id for t in truth) == Counter(t.class_id for t in predictions),
        "accepted": bool(accepted and not failed), "failed": bool(failed),
    }


def wilson(successes: int, count: int, z=1.959963984540054) -> list[float] | None:
    if count == 0:
        return None
    p = successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize(scores: list[dict]) -> dict:
    n = len(scores)
    successes = sum(s["hand_exact_match"] for s in scores)
    accepted = [s for s in scores if s["accepted"]]
    return {
        "images": n, "successes": successes,
        "hand_exact_match": successes / n if n else None,
        "wilson95_image_independence_assumption": wilson(successes, n),
        "model_exact": sum(s["model_exact"] for s in scores) / n if n else None,
        "multiset_exact": sum(s["multiset_exact"] for s in scores) / n if n else None,
        "accepted_coverage": len(accepted) / n if n else None,
        "accepted_exact": sum(s["model_exact"] for s in accepted) / len(accepted) if accepted else None,
        "failed_requests": sum(s["failed"] for s in scores),
        "note": "Independent-image Wilson interval; use source-cluster bootstrap when grouping is available.",
    }
