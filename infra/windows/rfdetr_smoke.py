"""Offline-cache RF-DETR constructor/prediction check; not an accuracy test."""
from __future__ import annotations
import importlib.metadata
import json
import os
from pathlib import Path

os.environ.setdefault("RF_HOME", r"D:\mahjong\models")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from PIL import Image
import torch
from rfdetr import RFDETRSmall

model = RFDETRSmall(pretrain_weights=str(Path(os.environ["RF_HOME"]) / "rf-detr-small.pth"), device="cuda")
prediction = model.predict(Image.new("RGB", (640, 400), (128, 128, 128)), threshold=0.99)
result = {"status": "passed", "rfdetr": importlib.metadata.version("rfdetr"), "torch": torch.__version__,
          "cuda": torch.cuda.is_available(), "model": "RFDETRSmall", "source_weights": "COCO pretraining only",
          "predicted_boxes_on_synthetic_image": len(prediction), "accuracy_evaluated": False}
output = Path(r"D:\mahjong\runs\infra-rfdetr-smoke")
output.mkdir(parents=True, exist_ok=True)
(output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result), flush=True)
