"""Validate CUDA, AMP, optimizer, torchvision ops and checkpoint resume locally.

This is a synthetic environment test, not a Mahjong model accuracy test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import torch
import torchvision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(20261025)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = torchvision.models.resnet18(weights=None, num_classes=43).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(2, 3, 224, 224, device="cuda")
    y = torch.tensor([0, 42], device="cuda")
    losses: list[float] = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        if not torch.isfinite(loss):
            raise AssertionError("Non-finite AMP loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    checkpoint = args.output / "smoke.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "step": 3}, checkpoint)
    state = torch.load(checkpoint, map_location="cuda", weights_only=True)
    restored = torchvision.models.resnet18(weights=None, num_classes=43).cuda()
    restored.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    model.eval()
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=torch.float32, device="cuda")
    kept = torchvision.ops.nms(boxes, torch.tensor([.9, .8, .7], device="cuda"), .5)
    if kept.tolist() != [0, 2]:
        raise AssertionError("Unexpected torchvision CUDA NMS result")
    torch.cuda.synchronize()
    result = {"status": "passed", "python": platform.python_version(), "torch": torch.__version__,
              "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "device_capability": list(torch.cuda.get_device_capability()),
              "losses": losses, "steps": 3, "checkpoint_resume": True, "cuda_nms": True,
              "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
              "elapsed_seconds": round(time.perf_counter()-started, 3)}
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
