"""Package one checkpoint with exact provenance; packaging is not release approval."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from ml.inference.manifest import class_ontology, validate_labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run = json.loads(args.run.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    with args.checkpoint.open("rb") as source:
        checkpoint_hash = hashlib.file_digest(source, "sha256").hexdigest()
    expected = {
        "checkpointSha256": checkpoint_hash,
        "runSha256": hashlib.sha256(args.run.read_bytes()).hexdigest(),
        "trainAnnotationSha256": run["annotationSha256"],
        "architecture": run["architecture"], "resolution": run["resolution"],
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Validation report does not match model provenance: {key}")
    if run.get("resumeSignature") is None or report.get("resumeSignatureSha256") != hashlib.sha256(json.dumps(run["resumeSignature"], sort_keys=True).encode()).hexdigest():
        raise ValueError("Validation report must verify the checkpoint-bound run signature")
    if report.get("partial", False):
        raise ValueError("Select the deployment threshold on the complete validation set")
    if run.get("validationAnnotationSha256") != report.get("validationAnnotationSha256"):
        raise ValueError("Validation annotations differ from the training run")
    if report.get("split") != "validation_only":
        raise ValueError("Threshold selection must come from validation only")
    if args.out.exists():
        raise FileExistsError("Use a new immutable model directory")
    checkpoint = args.out / "checkpoint.pth"
    class_map = Path("docs/data/class-map.json")
    labels = {}
    if class_map.exists():
        data = json.loads(class_map.read_text(encoding="utf-8"))
        for item in data.get("classes", []):
            labels[item["token"]] = item["label_zh"]
    manifest = {
        "version": args.version, "architecture": run["architecture"], "resolution": run["resolution"],
        "checkpoint": checkpoint.name, "checkpointSha256": checkpoint_hash, "checkpointBytes": args.checkpoint.stat().st_size,
        "classes": run["classes"], "ontology": class_ontology(run["classes"]), "labels": labels, "threshold": report["bestThreshold"], "acceptThreshold": report.get("acceptThreshold", .7),
        "dataAnnotationSha256": run["annotationSha256"], "codeCommit": run["codeCommit"],
        "productionValidated": False, "validationReport": "validation-report.json",
        "resumeSignatureSha256": report["resumeSignatureSha256"],
        "validationAnnotationSha256": report["validationAnnotationSha256"],
    }
    validate_labels(manifest)
    args.out.mkdir(parents=True)
    shutil.copy2(args.checkpoint, checkpoint)
    (args.out / "model.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(args.report, args.out / "validation-report.json")
    shutil.copy2(args.run, args.out / "run.json")


if __name__ == "__main__":
    main()
