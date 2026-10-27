"""Run one argv-based job with durable status/logs, independent of SSH.

Launch with start-job.ps1. Each run directory is immutable and cannot be reused.
Only this job's child process is waited on; unrelated processes are untouched.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    if spec.get("kind") == "offline_experiment_plan" or spec.get("launchable") is False:
        raise ValueError("Offline experiment plans are not launchable jobs; complete their post-boot preflight first")
    run_dir = Path(spec["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / "runner.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode("ascii"))
    os.close(fd)
    status_file = run_dir / "status.json"
    keep_awake = False
    if os.name == "nt":
        # Keep the system awake only while this runner is alive. This does not
        # alter the user's power plan or prevent display sleep.
        keep_awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    status = {"state": "starting", "runner_pid": os.getpid(), "started_at": now(), "argv": spec["argv"], "cwd": spec["cwd"],
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(), "keep_system_awake": keep_awake}
    write_status(status_file, status)
    try:
        with (run_dir / "stdout.log").open("ab", buffering=0) as stdout, (run_dir / "stderr.log").open("ab", buffering=0) as stderr:
            env = os.environ.copy()
            env.update({str(k): str(v) for k, v in spec.get("env", {}).items()})
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("PYTHONUTF8", "1")
            env.setdefault("PYTHONIOENCODING", "utf-8")
            process = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=env, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, shell=False)
            status.update(state="running", child_pid=process.pid)
            write_status(status_file, status)
            code = process.wait()
        status.update(state="succeeded" if code == 0 else "failed", exit_code=code, finished_at=now())
        write_status(status_file, status)
        return code
    except BaseException as exc:
        status.update(state="failed", error=str(exc), finished_at=now())
        write_status(status_file, status)
        with (run_dir / "stderr.log").open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        raise
    finally:
        if keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    raise SystemExit(main())
