"""Read durable queue progress and current epoch metrics; never controls jobs."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def epoch_metrics(run):
    path = run / "metrics.csv"
    epochs = {}
    if path.is_file():
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                value = row.get("epoch")
                if value:
                    epochs.setdefault(value, {}).update({k: v for k, v in row.items() if v})
    return [epochs[k] for k in sorted(epochs, key=float)]


def snapshot(root):
    state = read_json(root / "state.json")
    config = read_json(root / "config.json")
    receipts = []
    for job in config["experiments"]:
        file = root / "jobs" / (job["id"] + ".json")
        receipts.append(read_json(file) if file.exists() else {"id": job["id"], "state": "pending"})
    counts = dict(Counter(r["state"] for r in receipts))
    current = state.get("current_experiment")
    current_job = next((r for r in config["experiments"] if r["id"] == current), None)
    rows, metadata = [], {}
    if state.get("current_run_dir"):
        run = Path(state["current_run_dir"])
        rows = epoch_metrics(run)
        if (run / "run.json").exists():
            metadata = read_json(run / "run.json")
    unfinished = {r["id"] for r in receipts if r["state"] in {"pending", "starting", "running"}}
    estimate = sum(r["estimated_minutes"] for r in config["experiments"] if r["id"] in unfinished)
    if current_job and current in unfinished and rows and metadata.get("startedAt"):
        complete = sum(bool(r.get("train/loss")) for r in rows)
        maximum = metadata.get("config", {}).get("epochs")
        if complete and maximum:
            age = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(metadata["startedAt"])).total_seconds())
            current_remaining = max(0, maximum - complete) * age / complete / 60
            estimate += current_remaining - current_job["estimated_minutes"]
    return {"observed_at": datetime.now(timezone.utc).isoformat(), "queue": state, "counts": counts,
            "total": len(receipts), "plannedRemainingMinutes": round(estimate, 1),
            "currentEpochs": rows[-5:], "currentMaxEpochs": metadata.get("config", {}).get("epochs"),
            "experiments": [{"id": r["id"], "state": r["state"], "step": r.get("step"),
                             "error": r.get("error")} for r in receipts]}


def render(value):
    state = value["queue"]
    deadline = datetime.fromtimestamp(state["deadline_epoch_seconds"], timezone.utc).isoformat()
    counts = " ".join(f"{k}={v}" for k, v in sorted(value["counts"].items()))
    lines = [f"queue={state['queue_id']} state={state['state']} total={value['total']} {counts}",
             f"deadline_utc={deadline} planned_remaining_minutes={value['plannedRemainingMinutes']} (estimate, not guarantee)",
             f"current={state.get('current_experiment') or '-'} step={state.get('current_step') or '-'} child_pid={state.get('child_pid') or '-'}"]
    if state.get("wait_reason") and state["state"] == "waiting":
        lines.append("waiting=" + state["wait_reason"])
    if state.get("error"):
        lines.append("queue_error=" + state["error"])
    backgrounds = state.get("gpu_processes", state.get("wait_detail", {}))
    if isinstance(backgrounds, dict) and backgrounds.get("allowed_background_processes"):
        lines.append(f"verified_allowed_gpu_backgrounds={len(backgrounds['allowed_background_processes'])}")
    for gpu in state.get("gpu_sample", []):
        lines.append(f"gpu={gpu['gpu_index']} used_mib={gpu['memory_used_mib']}/{gpu['memory_total_mib']} utilization_percent={gpu['utilization_percent']}")
    for row in value["currentEpochs"]:
        fields = [f"epoch={int(float(row['epoch']))+1}/{value['currentMaxEpochs'] or '?'}"]
        for key, label in (("train/loss", "loss"), ("train/loss_ce", "classification"),
                           ("train/loss_bbox", "bbox"), ("val/mAP_50", "mAP50"), ("val/mAP_50_95", "mAP50_95")):
            metric_value = row.get(key)
            if metric_value is None or metric_value == "":
                metric_value = row.get({"val/mAP_50": "mAP50", "val/mAP_50_95": "mAP50_95"}.get(key, ""))
            if metric_value is not None and metric_value != "":
                fields.append(f"{label}={float(metric_value):.5f}")
        lines.append("  ".join(fields))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be >=1 second")
    previous = None
    try:
        while True:
            try:
                value = snapshot(args.queue)
                output = json.dumps(value, indent=2, ensure_ascii=False) if args.json else render(value)
                if output != previous:
                    if not args.json:
                        print(value["observed_at"], flush=True)
                    print(output, flush=True)
                    previous = output
                if not args.follow or value["queue"]["state"] in {"completed", "blocked", "interrupted"}:
                    return
            except (OSError, ValueError, csv.Error):
                if not args.follow:
                    raise
                print("Queue files are not ready or being updated; retrying...", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped watching. The background experiment queue is unchanged.", flush=True)


if __name__ == "__main__":
    main()
