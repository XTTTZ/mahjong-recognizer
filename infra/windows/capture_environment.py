"""Capture exact package versions and platform evidence without credentials."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")}
snapshot = {"captured_at": datetime.now(timezone.utc).isoformat(), "environment": Path(sys.prefix).name,
            "python_executable": sys.executable, "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower()))}
(args.output / "environment.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
lock = "--extra-index-url https://download.pytorch.org/whl/cu128\n" + "".join(f"{name}=={version}\n" for name, version in snapshot["packages"].items())
(args.output / "requirements-lock.txt").write_text(lock, encoding="utf-8")
check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
(args.output / "pip-check.txt").write_text(check.stdout + check.stderr, encoding="utf-8")
if check.returncode:
    raise RuntimeError("Dependency check failed; inspect pip-check.txt")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
print(json.dumps({k: v for k, v in snapshot.items() if k != "packages"}), flush=True)
