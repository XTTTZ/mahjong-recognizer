"""Compare deduplicated best/last checkpoints by complete-hand validation metrics."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

from ml.train.provenance import atomic_json, sha256
from ml.train.yolo_train import verify_receipt


def candidates(checkpoint, run):
    checkpoint = Path(checkpoint)
    run = json.loads(Path(run).read_text(encoding="utf-8"))
    if not run.get("finishedAt"):
        raise ValueError("Candidate evaluation requires a completed run")
    unique, result = set(), []
    for path in (checkpoint, checkpoint.with_name("last.pt")):
        if not path.is_file():
            continue
        verify_receipt(path, run)
        digest = sha256(path)
        if digest not in unique:
            result.append((path, digest))
            unique.add(digest)
    if not result:
        raise FileNotFoundError("No completed signed checkpoints found")
    return result


def ranking(report):
    group = report["selectionGroup"]
    selected = report["thresholds"][str(report["bestThreshold"])][group]
    return tuple(selected[key] or 0. for key in ("hand_exact_match", "model_exact", "accepted_coverage"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="Direct detector or generic detector run.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-run", type=Path)
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.classifier_run) != bool(args.classifier):
        raise ValueError("Cascade requires both classifier checkpoint and run")
    if args.out.exists():
        raise FileExistsError("Use a new immutable candidate-evaluation directory")
    detector_candidates = candidates(args.checkpoint, args.run)
    classifier_candidates = candidates(args.classifier, args.classifier_run) if args.classifier else [(None, None)]
    args.out.mkdir(parents=True)
    results = []
    for index, (detector, classifier) in enumerate(itertools.product(detector_candidates, classifier_candidates)):
        detector_path, detector_sha = detector
        classifier_path, classifier_sha = classifier
        destination = args.out / f"candidate-{index:02d}"
        if classifier_path:
            command = [sys.executable, "-m", "ml.eval.cascade_evaluate", "--dataset", str(args.dataset),
                "--detector", str(detector_path), "--detector-run", str(args.run),
                "--classifier", str(classifier_path), "--classifier-run", str(args.classifier_run), "--out", str(destination)]
        else:
            command = [sys.executable, "-m", "ml.eval.yolo_evaluate", "--dataset", str(args.dataset),
                "--checkpoint", str(detector_path), "--run", str(args.run), "--out", str(destination)]
        receipt = {"detector": str(detector_path), "detectorSha256": detector_sha,
                   "classifier": str(classifier_path) if classifier_path else None,
                   "classifierSha256": classifier_sha, "argv": command}
        with (args.out / f"candidate-{index:02d}.stdout.log").open("wb") as stdout, (args.out / f"candidate-{index:02d}.stderr.log").open("wb") as stderr:
            code = subprocess.run(command, stdout=stdout, stderr=stderr, shell=False).returncode
        report_path = destination / "report.json"
        receipt.update(exitCode=code, reportPath=str(report_path))
        if code == 0 and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("partial") or report.get("split") != "validation_only" or report["checkpointSha256"] != detector_sha:
                raise ValueError("Candidate evaluator returned an inconsistent report")
            if classifier_path and report.get("classifierCheckpointSha256") != classifier_sha:
                raise ValueError("Candidate classifier pairing changed during evaluation")
            receipt.update(state="succeeded", ranking=ranking(report))
        else:
            receipt.update(state="failed")
        results.append(receipt)
        atomic_json(args.out / "selection.json", {"candidates": results, "trainingStarted": False})
        print(json.dumps({key: value for key, value in receipt.items() if key != "argv"}), flush=True)
    successful = [receipt for receipt in results if receipt["state"] == "succeeded"]
    if not successful:
        raise RuntimeError("No candidate completed valid full-image evaluation")
    winner = max(successful, key=lambda receipt: tuple(receipt["ranking"]))
    report = json.loads(Path(winner["reportPath"]).read_text(encoding="utf-8"))
    report.update(selectedCheckpoint=winner["detector"], selectedClassifier=winner["classifier"],
        candidateSelection="best and last deduplicated by SHA256; full-image complete-hand correctness, model exact, then coverage",
        candidateReports=results)
    atomic_json(args.out / "report.json", report)
    print(json.dumps({"selectedCheckpoint": winner["detector"], "selectedClassifier": winner["classifier"], "report": str(args.out / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
