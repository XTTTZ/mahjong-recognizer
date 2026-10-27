"""Select confidence on validation only, using the same output policy as the API."""
import argparse
from collections import defaultdict
import json
import hashlib
from pathlib import Path
import statistics
import time

from ml.eval.metrics import Tile, score_image, summarize
from ml.eval.operating_point import choose_threshold, score_predictions
from ml.train.provenance import check_evaluation_data, dataset_contract, sha256
from ml.train.rfdetr_train import checkpoint_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.15,0.25,0.35,0.5,0.65")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--accept-threshold", type=float, default=0.7)
    args = parser.parse_args()
    thresholds = sorted({float(v) for v in args.thresholds.split(",")})
    if not thresholds or not all(0 < v < 1 for v in thresholds) or not 0 < args.accept_threshold < 1:
        raise ValueError("Thresholds must be in (0,1)")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    from PIL import Image
    from rfdetr import RFDETR
    import torch
    metadata = json.loads(args.run.read_text(encoding="utf-8"))
    contract = dataset_contract(args.dataset, verify_images=True)
    check_evaluation_data(metadata, contract)
    checkpoint_sha256 = sha256(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint_contract(saved, metadata["resumeSignature"], require_training_state=False)
    expected_type = {"rfdetr-small": "RFDETRSmall", "rfdetr-medium": "RFDETRMedium", "rfdetr-large": "RFDETRLarge"}[metadata["architecture"]]
    if saved.get("model_name") != expected_type or saved.get("model_config", {}).get("resolution") != metadata["resolution"]:
        raise ValueError("Checkpoint architecture/resolution differs from training run")
    del saved
    classes = metadata["classes"]
    dataset = args.dataset / "valid"
    coco = json.loads((dataset / "_annotations.coco.json").read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    annotations = defaultdict(list)
    for a in coco["annotations"]:
        annotations[a["image_id"]].append(Tile(names[a["category_id"]], tuple(a["bbox"])))
    model = RFDETR.from_checkpoint(str(args.checkpoint), resolution=metadata["resolution"], device="cuda")
    if type(model).__name__ != expected_type or model.model_config.num_classes != len(classes):
        raise ValueError("Checkpoint type/classifier differs from training run")
    if sha256(args.checkpoint) != checkpoint_sha256:
        raise ValueError("Checkpoint changed during loading; evaluate an immutable candidate")
    saved_names = getattr(model.model, "class_names", None)
    if saved_names and list(saved_names) != [classes[str(i)] for i in range(len(classes))]:
        raise ValueError("Checkpoint class names differ from run.json")
    if args.out.exists():
        raise FileExistsError("Use a new immutable evaluation directory")
    args.out.mkdir(parents=True)
    buckets = {name: {threshold: [] for threshold in thresholds}
               for name in ("all", "single_tile", "ten_or_more_tiles", "thirteen_or_fourteen_tiles", "raw_ten_or_more_tiles", "oracle_roi_ten_or_more_tiles")}
    latencies, failed = [], 0
    records = coco["images"][:args.limit]
    with (args.out / "validation-predictions.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            # Reject inconsistent truth rather than silently rotate only the image.
            with Image.open(dataset / record["file_name"]) as source:
                if source.size != (record["width"], record["height"]) or source.getexif().get(274, 1) != 1:
                    raise ValueError("Normalize EXIF and truth coordinates in the data derivative first")
                image = source.convert("RGB")
            raw, error_type = [], None
            start = time.perf_counter()
            try:
                prediction = model.predict(image, threshold=min(thresholds))
                for box, confidence, class_id in zip(prediction.xyxy, prediction.confidence, prediction.class_id):
                    x1, y1, x2, y2 = map(float, box)
                    raw.append({"classId": classes[str(int(class_id))], "bbox": [x1, y1, x2-x1, y2-y1], "score": float(confidence)})
                # Output validation is inside this boundary: malformed model output
                # is a failed request, never a dropped image from the denominator.
                image_scores = {t: score_predictions(annotations[record["id"]], raw, image.width, image.height, t, args.accept_threshold) for t in thresholds}
            except Exception as error:
                failed += 1
                error_type = type(error).__name__
                image_scores = {t: score_image(annotations[record["id"]], [], accepted=False, failed=True) for t in thresholds}
            latencies.append((time.perf_counter() - start) * 1000)
            count = len(annotations[record["id"]])
            selected = ["all"] + (["single_tile"] if count == 1 else []) + (["ten_or_more_tiles"] if count >= 10 else []) + (["thirteen_or_fourteen_tiles"] if count in (13, 14) else [])
            if count >= 10:
                selected.append("oracle_roi_ten_or_more_tiles" if record.get("oracle_derived", False) else "raw_ten_or_more_tiles")
            for name in selected:
                for threshold, score in image_scores.items():
                    buckets[name][threshold].append(score)
            output.write(json.dumps({"image": record, "predictions": raw, "truthCount": count,
                                     "failed": error_type is not None, "errorType": error_type}) + "\n")
            if len(latencies) % 25 == 0:
                print(json.dumps({"event": "evaluated", "images": len(latencies), "failed": failed}), flush=True)
    reports = {str(t): {name: summarize(scores[t]) for name, scores in buckets.items()} for t in thresholds}
    group = "raw_ten_or_more_tiles" if buckets["raw_ten_or_more_tiles"][thresholds[0]] else "ten_or_more_tiles" if buckets["ten_or_more_tiles"][thresholds[0]] else "all"
    report = {"checkpoint": str(args.checkpoint), "split": "validation_only", "externalPhoneBenchmark": False,
              "partial": len(records) != len(coco["images"]), "selectionGroup": group,
              "checkpointSha256": checkpoint_sha256, "runSha256": sha256(args.run),
              "trainAnnotationSha256": metadata["annotationSha256"],
              "resumeSignatureSha256": hashlib.sha256(json.dumps(metadata["resumeSignature"], sort_keys=True).encode()).hexdigest(),
              "dataContract": contract,
              "validationAnnotationSha256": contract["splits"]["valid"]["annotationSha256"],
              "architecture": metadata["architecture"], "resolution": metadata["resolution"],
              "acceptThreshold": args.accept_threshold, "bestThreshold": choose_threshold(reports, thresholds, group),
              "thresholds": reports, "inferenceFailedImages": failed,
              "meanInferenceMsIncludingFirst": statistics.mean(latencies),
              "medianInferenceMsIncludingFirst": statistics.median(latencies),
              "timingScope": "local prediction and output normalization; excludes model load, upload and Modal cold start",
              "note": "Research validation with API rejection policy. Does not establish real-room generalization or release readiness."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
