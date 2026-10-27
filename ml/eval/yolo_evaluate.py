"""Validation-only full-image YOLO/cascade evaluation with API rejection policy."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

from ml.eval.metrics import Tile, score_image, summarize
from ml.eval.operating_point import choose_threshold, score_predictions
from ml.inference.postprocess import normalize_tiles
from ml.train.provenance import check_evaluation_data, dataset_contract, sha256
from ml.train.yolo_train import verify_receipt


def evaluate(predictor, dataset, out, metadata, thresholds=(.15, .25, .35, .5, .65), accept_threshold=.7, limit=None,
             split="valid", select_threshold=True, expected_annotation_sha256=None):
    from PIL import Image
    from ml.eval.coco_metrics import coco_metrics
    dataset, out = Path(dataset), Path(out)
    if out.exists():
        raise FileExistsError("Use an immutable new validation output directory")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not thresholds or not all(0 < value < 1 for value in thresholds) or not 0 < accept_threshold < 1:
        raise ValueError("Invalid operating point grid")
    if split not in {"valid", "test"}:
        raise ValueError("Only validation or explicitly locked sealed test is supported")
    if split == "test" and (select_threshold or len(thresholds) != 1 or not metadata.get("winnerLockSha256") or not expected_annotation_sha256 or limit is not None):
        raise ValueError("Sealed test requires one frozen operating point, a winner lock, full coverage and annotation hash")
    annotation_path = dataset / split / "_annotations.coco.json"
    if expected_annotation_sha256 and sha256(annotation_path) != expected_annotation_sha256:
        raise ValueError("Evaluation annotation bytes differ from frozen selection")
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    tokens = {row["id"]: row["name"] for row in coco["categories"]}
    category_ids = {token: index for index, token in tokens.items()}
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(Tile(tokens[annotation["category_id"]], tuple(annotation["bbox"])))
    names = ("all", "single_tile", "ten_or_more_tiles", "thirteen_or_fourteen_tiles", "raw_ten_or_more_tiles", "oracle_roi_ten_or_more_tiles")
    buckets = {name: {threshold: [] for threshold in thresholds} for name in names}
    latencies, coco_predictions, failures = [], [], 0
    records = coco["images"][:limit]
    out.mkdir(parents=True)
    with (out / "validation-predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            with Image.open(dataset / split / record["file_name"]) as source:
                image = source.convert("RGB")
            started = time.perf_counter()
            raw, error_type = [], None
            try:
                raw = predictor.predict(image)
                scores = {threshold: score_predictions(annotations[record["id"]], raw,
                    image.width, image.height, threshold, accept_threshold) for threshold in thresholds}
                normalized, _ = normalize_tiles(raw, image.width, image.height, accept_threshold)
                coco_predictions.extend({"image_id": record["id"], "category_id": category_ids[tile["classId"]],
                    "bbox": tile["bbox"], "score": tile["score"]} for tile in normalized)
            except Exception as error:
                failures += 1
                error_type = type(error).__name__
                scores = {threshold: score_image(annotations[record["id"]], [], accepted=False, failed=True) for threshold in thresholds}
            latencies.append((time.perf_counter()-started)*1000)
            count = len(annotations[record["id"]])
            selected = ["all"] + (["single_tile"] if count == 1 else []) + (["ten_or_more_tiles"] if count >= 10 else []) + (["thirteen_or_fourteen_tiles"] if count in (13, 14) else [])
            if count >= 10:
                selected.append("oracle_roi_ten_or_more_tiles" if record.get("oracle_derived", False) else "raw_ten_or_more_tiles")
            for name in selected:
                for threshold in thresholds:
                    buckets[name][threshold].append(scores[threshold])
            stream.write(json.dumps({"image": record, "predictions": raw, "truthCount": count,
                "failed": error_type is not None, "errorType": error_type}) + "\n")
            stream.flush()
            if len(latencies) % 25 == 0:
                print(json.dumps({"event": "evaluated", "images": len(latencies), "failed": failures}), flush=True)
    reports = {str(threshold): {name: summarize(value[threshold]) for name, value in buckets.items()} for threshold in thresholds}
    group = "raw_ten_or_more_tiles" if buckets["raw_ten_or_more_tiles"][thresholds[0]] else "ten_or_more_tiles" if buckets["ten_or_more_tiles"][thresholds[0]] else "all"
    selected_ids = {image["id"] for image in records}
    report = dict(metadata, split="validation_only" if split == "valid" else "sealed_test_only", partial=len(records) != len(coco["images"]),
        externalPhoneBenchmark=False, productionValidated=False, selectionGroup=group,
        bestThreshold=choose_threshold(reports, thresholds, group) if select_threshold else thresholds[0], acceptThreshold=accept_threshold,
        operatingPointLocked=not select_threshold, testUsedForSelection=False,
        thresholds=reports, inferenceFailedImages=failures,
        meanInferenceMsIncludingFirst=statistics.mean(latencies), medianInferenceMsIncludingFirst=statistics.median(latencies),
        timingScope="local detector + crop classifier + output normalization, excludes model load/network/cold start",
        cocoAP=coco_metrics(records, [a for a in coco["annotations"] if a["image_id"] in selected_ids], coco_predictions, metadata["dataContract"]["classes"]),
        note="Research validation; classification-only oracle crop accuracy is never the complete-hand result",
        completedAt=datetime.now(timezone.utc).isoformat())
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def run_metadata(run_path, checkpoint, contract):
    run_path, checkpoint = Path(run_path), Path(checkpoint)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if not run.get("finishedAt"):
        raise ValueError("Evaluate an immutable completed training run")
    check_evaluation_data(run, contract)
    verify_receipt(checkpoint, run)
    return run, {"checkpointSha256": sha256(checkpoint), "runSha256": sha256(run_path),
        "trainAnnotationSha256": contract["splits"]["train"]["annotationSha256"],
        "validationAnnotationSha256": contract["splits"]["valid"]["annotationSha256"],
        "resumeSignatureSha256": hashlib.sha256(json.dumps(run["resumeSignature"], sort_keys=True).encode()).hexdigest(),
        "dataContract": contract, "architecture": run["architecture"], "resolution": run["resolution"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    contract = dataset_contract(args.dataset, verify_images=True)
    run, metadata = run_metadata(args.run, args.checkpoint, contract)
    if run.get("generic"):
        raise ValueError("One-class detector is a cascade component; evaluate full hands with cascade_evaluate")
    from ml.inference.yolo_model import RawYOLO
    model = RawYOLO(args.checkpoint, run["classes"], run["resolution"], threshold=.05)
    metadata.update(predictionFloor=.05, nmsIou=.6)
    evaluate(model, args.dataset, args.out, metadata, limit=args.limit)


if __name__ == "__main__":
    main()
