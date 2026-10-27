"""Package validation-bound YOLO/cascade weights; packaging never promotes a model."""
import argparse
import json
from pathlib import Path
import shutil

from ml.inference.manifest import validate_labels
from ml.train.provenance import sha256
from ml.train.yolo_train import verify_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Direct YOLO run or cascade detector run.json")
    parser.add_argument("--checkpoint", type=Path, required=True, help="YOLO checkpoint or cascade detector checkpoint")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--classifier-run", type=Path)
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strip-classifier-training-state", action="store_true",
                        help="Losslessly remove classifier optimizer/scheduler payload; keeps exact model tensors")
    args = parser.parse_args()
    run = json.loads(args.run.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    verify_receipt(args.checkpoint, run)
    if report.get("split") != "validation_only" or report.get("partial") or report.get("runSha256") != sha256(args.run) or report["checkpointSha256"] != sha256(args.checkpoint):
        raise ValueError("Package only the exact completed full-validation candidate")
    if report["dataContract"]["classes"] != run["classes"] or report["dataContract"]["splits"] != run["resumeSignature"]["data"]:
        raise ValueError("Validation contract differs from training")
    manifest = {"architecture": report["architecture"], "version": args.version,
        "classes": run["classes"], "ontology": "faces42", "resolution": run["resolution"],
        "threshold": report["bestThreshold"], "acceptThreshold": report["acceptThreshold"],
        "nmsIou": report["nmsIou"], "productionValidated": False,
        "validationReportSha256": sha256(args.report), "sourceRunSha256": sha256(args.run),
        "evaluationDevice": report.get("evaluationDevice"),
        "sourceTrainingStatus": report.get("sourceTrainingStatus"),
        "sourceClassifierTrainingStatus": report.get("sourceClassifierTrainingStatus"),
        "recoveryNotTrainingCompletion": report.get("recoveryNotTrainingCompletion", False),
        "sourceCheckpointSha256": sha256(args.checkpoint),
        "runtime": {"ultralytics": "8.4.189", "torch": "2.8.0", "torchvision": "0.23.0"},
        "license": "Ultralytics AGPL-3.0; evaluate licensing before public deployment"}
    files = {"weights.pt": args.checkpoint}
    if report["architecture"] == "yolo-efficientnet-v2-s-cascade":
        if not args.classifier or not args.classifier_run or report.get("classifierCheckpointSha256") != sha256(args.classifier) or report.get("classifierRunSha256") != sha256(args.classifier_run):
            raise ValueError("Cascade package must preserve the exact evaluated classifier pairing")
        classifier = json.loads(args.classifier_run.read_text(encoding="utf-8"))
        verify_receipt(args.classifier, classifier)
        if classifier["classes"] != run["classes"] or classifier["resumeSignature"]["data"] != run["resumeSignature"]["data"]:
            raise ValueError("Cascade stages use different class/data contracts")
        if classifier["config"]["resolution"] != report["classifierResolution"] or classifier["config"]["context"] != report["cropContext"]:
            raise ValueError("Cascade report crop geometry differs from the signed classifier")
        manifest.update(format="two-stage-pt", classifierResolution=report["classifierResolution"],
            cropContext=report["cropContext"], detectorThreshold=report["detectorThreshold"],
            scoreFusion=report["scoreFusion"], classifierSourceRunSha256=sha256(args.classifier_run),
            detectorArchitecture=run["architecture"],classifierArchitecture=classifier["architecture"])
        files = {"detector.pt": args.checkpoint, "classifier.pt": args.classifier}
        from ml.inference.cascade_model import verify_model_directory
    else:
        if report["architecture"] not in {"yolov8s", "yolo26s", "yolo26m"}:
            raise ValueError("Unsupported direct YOLO architecture")
        manifest.update(format="ultralytics-pt")
        from ml.inference.yolo_model import verify_model_directory
    manifest["filesSha256"] = {name: sha256(path) for name, path in files.items()}
    validate_labels(manifest)
    if args.out.exists():
        raise FileExistsError("Use an immutable new model package directory")
    args.out.mkdir(parents=True)
    for name, path in files.items():
        if name == "classifier.pt" and args.strip_classifier_training_state:
            import torch
            saved = torch.load(path, map_location="cpu", weights_only=True)
            exported = {key: saved[key] for key in ("model", "classes", "config", "signatureSha256", "epoch")}
            torch.save(exported, args.out / name)
            reread = torch.load(args.out / name, map_location="cpu", weights_only=True)
            if set(reread["model"]) != set(saved["model"]) or any(not torch.equal(saved["model"][key], reread["model"][key]) for key in saved["model"]):
                raise ValueError("Lossless classifier export changed model tensors")
            manifest["classifierExport"] = {"trainingStateRemoved": True, "allModelTensorsExactlyEqual": True,
                "sourceCheckpointSha256": sha256(path), "sourceBytes": path.stat().st_size,
                "exportedBytes": (args.out / name).stat().st_size}
        else:
            shutil.copyfile(path, args.out / name)
        manifest["filesSha256"][name] = sha256(args.out / name)
    shutil.copyfile(args.report, args.out / "validation-report.json")
    (args.out / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify_model_directory(args.out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
