"""End-to-end paired-stage evaluation on actual detector boxes, never oracle boxes."""
import argparse
from pathlib import Path

from ml.eval.yolo_evaluate import evaluate, run_metadata
from ml.train.provenance import dataset_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--detector-run", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--classifier-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    contract = dataset_contract(args.dataset, verify_images=True)
    detector_run, metadata = run_metadata(args.detector_run, args.detector, contract)
    classifier_run, classifier_metadata = run_metadata(args.classifier_run, args.classifier, contract)
    if not detector_run.get("generic") or classifier_run["architecture"] != "efficientnet-v2-s":
        raise ValueError("Cascade requires a one-class detector and the reviewed classifier")
    from ml.inference.yolo_model import RawYOLO
    from ml.inference.cascade_model import RawCascade
    detector = RawYOLO(args.detector, contract["classes"], detector_run["resolution"], threshold=.05, generic=True)
    config = classifier_run["config"]
    model = RawCascade(detector, args.classifier, contract["classes"], config["resolution"], config["context"])
    metadata.update(architecture="yolo-efficientnet-v2-s-cascade", detectorCheckpointSha256=metadata["checkpointSha256"],
        classifierCheckpointSha256=classifier_metadata["checkpointSha256"], classifierRunSha256=classifier_metadata["runSha256"],
        classifierResumeSignatureSha256=classifier_metadata["resumeSignatureSha256"], classifierResolution=config["resolution"],
        cropContext=config["context"], detectorThreshold=.05, nmsIou=.6, scoreFusion="detector_probability_times_classifier_probability",
        localizationScope="predicted_detector_boxes", classifierDataContract=contract)
    evaluate(model, args.dataset, args.out, metadata, limit=args.limit)


if __name__ == "__main__":
    main()
