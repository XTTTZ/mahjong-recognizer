"""Stop one owned training tree after its first complete readable checkpoint.

Used for the annotation-scope review. It never targets another run or resumes
training. A separate pause record distinguishes this from a training failure.
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--timeout", type=int, default=1200)
args = parser.parse_args()
checkpoint = args.run / "checkpoint_0.ckpt"
started = time.monotonic()
while time.monotonic() - started < args.timeout:
    status = json.loads((args.run / "status.json").read_text(encoding="utf-8"))
    if status["state"] != "running":
        raise RuntimeError(f"Training is already {status['state']}; inspect it manually")
    if checkpoint.exists() and checkpoint.stat().st_size > 1_000_000:
        previous = checkpoint.stat().st_size
        time.sleep(3)
        if checkpoint.stat().st_size == previous:
            validation = args.run / "checkpoint-validation.json"
            check = subprocess.run([sys.executable, str(Path(__file__).with_name("validate_checkpoint.py")), str(checkpoint), "--output", str(validation)], capture_output=True, text=True)
            if check.returncode == 0:
                child = int(status["child_pid"])
                script = f"Get-CimInstance Win32_Process -Filter 'ProcessId={child}' | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
                command = subprocess.run(["powershell", "-NoProfile", "-EncodedCommand", base64.b64encode(script.encode("utf-16le")).decode()], capture_output=True, text=True, check=True)
                process = json.loads(command.stdout)
                if args.run.name not in process.get("CommandLine", ""):
                    raise RuntimeError("Process command no longer matches this run; refusing to terminate")
                pause = {"status": "stopped_after_checkpoint_for_annotation_scope_review", "requestedBy": "root", "stoppedAt": datetime.now(timezone.utc).isoformat(),
                         "reason": "Broadcast screenshots may annotate only the bottom hand while other visible tiles remain unlabeled. Research baseline only; no promotion or deployment.",
                         "child_pid": child, "resumeCheckpoint": str(checkpoint), "checkpointValidation": json.loads(validation.read_text()),
                         "sealedTestUsed": False, "productionValidated": False}
                result = subprocess.run(["taskkill", "/PID", str(child), "/T", "/F"], capture_output=True, text=True)
                pause["terminationExitCode"] = result.returncode
                pause["terminationOutput"] = result.stdout + result.stderr
                (args.run / "annotation-review-pause.json").write_text(json.dumps(pause, indent=2), encoding="utf-8")
                print(json.dumps(pause), flush=True)
                if result.returncode:
                    raise RuntimeError("Owned tree termination failed; inspect pause record")
                break
    time.sleep(2)
else:
    raise TimeoutError("No complete first-epoch checkpoint found; training left untouched")
