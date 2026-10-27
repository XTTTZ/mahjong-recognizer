"""Short durable task used to test that a job survives SSH disconnection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
for step in range(4):
    (args.output / "checkpoint.json").write_text(json.dumps({"step": step, "complete": False}), encoding="utf-8")
    print(f"step={step}", flush=True)
    time.sleep(5)
(args.output / "checkpoint.json").write_text(json.dumps({"step": 4, "complete": True}), encoding="utf-8")
print("finished", flush=True)
