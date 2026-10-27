"""Score exactly the detections and rejection policy served by the HTTP API."""
from ml.eval.metrics import Tile, score_image
from ml.inference.postprocess import MAX_DETECTIONS, normalize_tiles


def score_predictions(truth, raw, width, height, threshold, accept_threshold):
    retained = [t for t in raw if t["score"] > threshold]
    # The API returns 422 for a crowded scene instead of a successful result.
    if len(retained) > MAX_DETECTIONS:
        return score_image(truth, [], accepted=False, failed=True)
    tiles, reasons = normalize_tiles(retained, width, height, accept_threshold)
    return score_image(truth, [Tile(t["classId"], tuple(t["bbox"])) for t in tiles], accepted=not reasons)


def choose_threshold(reports, thresholds, group):
    return max(thresholds, key=lambda t: tuple(reports[str(t)][group][key] or 0.0
        for key in ("hand_exact_match", "model_exact", "accepted_coverage")))
