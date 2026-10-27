"""Read a background run's state and merged epoch metrics; never controls the job."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def snapshot(run, history=8):
    status_file = run / "status.json"
    status = json.loads(status_file.read_text(encoding="utf-8-sig")) if status_file.exists() else {"state": "waiting"}
    rows = {}
    metrics = run / "metrics.csv"
    if metrics.exists():
        with metrics.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                epoch = row.get("epoch", "")
                if epoch:
                    rows.setdefault(epoch, {}).update({k: v for k, v in row.items() if v})
    run_file = run / "run.json"
    metadata = json.loads(run_file.read_text(encoding="utf-8-sig")) if run_file.exists() else {}
    maximum = metadata.get("config", {}).get("epochs")
    last = sorted(rows, key=lambda value: float(value))[-history:]
    lines = [f"run={run.name} state={status['state']} child_pid={status.get('child_pid', '-')} max_epochs={maximum or '-'}"]
    for epoch in last:
        row = rows[epoch]
        fields = [f"epoch={int(float(epoch))+1}", f"step={row.get('step', '-')}"]
        for key, label in (("train/loss", "loss"), ("train/loss_ce", "classification"),
                           ("train/loss_bbox", "bbox"), ("train/loss_giou", "giou"),
                           ("val/mAP_50", "mAP50"), ("val/mAP_50_95", "mAP50_95")):
            if row.get(key):
                fields.append(f"{label}={float(row[key]):.5f}")
        lines.append("  ".join(fields))
    checkpoints = sorted(run.glob("checkpoint_*.ckpt"), key=lambda path: path.stat().st_mtime)
    if checkpoints:
        latest = checkpoints[-1]
        lines.append(f"checkpoint={latest.name} bytes={latest.stat().st_size}")
    if status.get("exit_code") is not None:
        lines.append(f"exit_code={status['exit_code']} finished_at={status.get('finished_at', '-')}")
    if not rows:
        lines.append("Epoch metrics are not available yet. See stdout.log / stderr.log for startup or batch logs.")
    return "\n".join(lines), status["state"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be >=1 second")
    previous = None
    try:
        while True:
            try:
                output, state = snapshot(args.run)
                if output != previous:
                    print(datetime.now(timezone.utc).isoformat(), flush=True)
                    print(output, flush=True)
                    previous = output
                if not args.follow or state in {"succeeded", "failed"}:
                    return
            except (OSError, ValueError, csv.Error):
                if not args.follow:
                    raise
                print("Metrics are being updated; retrying...", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped watching. The background training job is unchanged.", flush=True)


if __name__ == "__main__":
    main()
