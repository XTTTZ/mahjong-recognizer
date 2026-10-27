"""Verify a trusted locally-produced Lightning checkpoint before stopping a job."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
if not checkpoint.get("state_dict") or not checkpoint.get("optimizer_states"):
    raise RuntimeError("Checkpoint lacks model or optimizer state")
if checkpoint.get("epoch", -1) < 0 or checkpoint.get("global_step", 0) <= 0:
    raise RuntimeError("Checkpoint does not contain a completed training epoch")
with args.checkpoint.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
result = {"path": str(args.checkpoint), "size_bytes": args.checkpoint.stat().st_size, "sha256": digest,
          "epoch_zero_based": checkpoint["epoch"], "global_step": checkpoint["global_step"],
          "model_tensor_entries": len(checkpoint["state_dict"]),
          "optimizer_states": len(checkpoint["optimizer_states"]),
          "lr_scheduler_states": len(checkpoint.get("lr_schedulers", [])), "fully_readable": True}
args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result), flush=True)
