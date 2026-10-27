"""Compare a bounded set of completed RF-DETR checkpoints on validation only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from ml.train.provenance import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-checkpoints", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.max_checkpoints <= 5:
        parser.error("max-checkpoints must be1..5")
    run = args.run_dir / "run.json"
    metadata = json.loads(run.read_text(encoding="utf-8"))
    if not metadata.get("finishedAt"):
        raise ValueError("Select from a completed run, not a still-changing training directory")
    if args.out.exists():
        raise FileExistsError("Use a new immutable selection directory")
    args.out.mkdir(parents=True)
    archives = sorted(args.run_dir.glob("checkpoint_*.ckpt"), key=lambda p: p.stat().st_mtime)
    order = [args.run_dir / "checkpoint_best_total.pth", args.run_dir / "last.ckpt",
             args.run_dir / "checkpoint_best_ema.pth"]
    if archives:
        order += [archives[-1], archives[len(archives)//2]]
    candidates, seen = [], set()
    for path in order:
        if not path.is_file():
            continue
        digest = sha256(path)
        if digest not in seen:
            candidates.append((path, digest))
            seen.add(digest)
        if len(candidates) >= args.max_checkpoints:
            break
    reports, failures = [], []
    for index, (checkpoint, digest) in enumerate(candidates):
        output = args.out / f"candidate-{index+1:02d}"
        command = [sys.executable, "-m", "ml.eval.rfdetr_evaluate", "--run", str(run),
                   "--checkpoint", str(checkpoint), "--dataset", str(args.dataset), "--out", str(output)]
        print(json.dumps({"event": "evaluate_candidate", "checkpoint": checkpoint.name, "sha256": digest}), flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures.append({"checkpoint": str(checkpoint), "sha256": digest, "exitCode": result.returncode})
        else:
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            reports.append((report, output / "report.json"))
    if not reports:
        atomic_json(args.out / "failure.json", {"evaluated": len(candidates), "failures": failures})
        raise RuntimeError("No candidate produced a valid complete validation report")

    def rank(entry):
        report, _ = entry
        scores = report["thresholds"][str(report["bestThreshold"])][report["selectionGroup"]]
        return tuple(scores[key] or 0.0 for key in ("hand_exact_match", "model_exact", "accepted_coverage"))

    winner, path = max(reports, key=rank)
    winner = dict(winner, selectionReport=str(path), productionValidated=False,
                  selectionPolicy="validation automatic complete-hand exact, then model exact, then coverage",
                  candidates=[{"checkpoint": r["checkpoint"], "checkpointSha256": r["checkpointSha256"],
                               "report": str(p), "score": rank((r, p))} for r, p in reports],
                  candidateFailures=failures)
    atomic_json(args.out / "report.json", winner)
    print(json.dumps({"event": "candidate_selection_finished", "report": str(args.out / "report.json"),
                      "checkpoint": winner["checkpoint"], "scores": rank((winner, path))}), flush=True)


if __name__ == "__main__":
    main()
