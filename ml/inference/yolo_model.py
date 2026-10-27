"""Ultralytics prediction adapter. Imports GPU libraries only at construction."""
import json
from pathlib import Path

from ml.inference.manifest import file_sha256, model_file, validate_labels


def result_tiles(result, classes, generic=False):
    tiles = []
    for xyxy, score, category in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
        x1, y1, x2, y2 = map(float, xyxy)
        if int(category) != category or int(category) not in range(1 if generic else len(classes)):
            raise ValueError("Unexpected detector class index")
        token = "tile" if generic else classes[str(int(category))]
        tiles.append({"classId": token, "label": token, "bbox": [x1, y1, x2-x1, y2-y1], "score": float(score)})
    return tiles


def verify_model_directory(model_dir):
    directory = Path(model_dir)
    manifest = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    if manifest.get("architecture") not in {"yolov8s", "yolo26s", "yolo26m"} or manifest.get("format") != "ultralytics-pt":
        raise ValueError("Unexpected YOLO architecture/format")
    if set(manifest.get("filesSha256", {})) != {"weights.pt"}:
        raise ValueError("Packaged YOLO must hash its weights")
    if file_sha256(model_file(directory, "weights.pt")) != manifest["filesSha256"]["weights.pt"]:
        raise ValueError("YOLO weights differ from manifest")
    validate_labels(manifest)
    return directory, manifest


class RawYOLO:
    def __init__(self, checkpoint, classes, resolution=960, device="cuda", threshold=.05, generic=False, iou=.6):
        import torch
        from ultralytics import YOLO
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing implicit CPU inference")
        if not Path(checkpoint).is_file():
            raise FileNotFoundError("YOLO weights must be cached before execution")
        self.model = YOLO(str(checkpoint), task="detect")
        self.classes, self.resolution, self.device = classes, resolution, 0 if device == "cuda" else device
        self.threshold, self.generic, self.iou = threshold, generic, iou
        expected = ["tile"] if generic else [classes[str(i)] for i in range(len(classes))]
        if list(self.model.names.values()) != expected:
            raise ValueError("YOLO checkpoint label mapping differs from the signed run")

    def predict(self, image):
        result = self.model.predict(image.convert("RGB"), imgsz=self.resolution, device=self.device,
            conf=self.threshold, iou=self.iou, max_det=300, quantize=16 if self.device == 0 else 32,
            verbose=False, augment=False, rect=False)[0]
        return result_tiles(result, self.classes, self.generic)


class YOLOPredictor(RawYOLO):
    def __init__(self, model_dir, device="cuda"):
        directory, self.manifest = verify_model_directory(model_dir)
        super().__init__(directory / "weights.pt", self.manifest["classes"], self.manifest["resolution"],
            device, self.manifest["threshold"], iou=self.manifest.get("nmsIou", .6))
        self.version = self.manifest["version"]
