"""Hugging Face RT-DETRv2 adapter with a saved processor and explicit class map.

This module does not import torch/transformers until constructing the predictor,
so data-contract and metadata checks do not require a GPU runtime.
"""
from __future__ import annotations

import json
from pathlib import Path
from ml.inference.manifest import file_sha256, model_file, validate_labels


def prepare_input(image, target=None, *, resolution=640, mode="warp"):
    """Optional letterbox shared exactly by training, validation and inference."""
    if mode == "warp":
        return image, target, None
    if mode != "letterbox":
        raise ValueError("Unsupported input mode")
    from PIL import Image
    width, height = image.size
    factor = min(resolution / width, resolution / height)
    resized_w, resized_h = max(1, round(width * factor)), max(1, round(height * factor))
    left, top = (resolution - resized_w) // 2, (resolution - resized_h) // 2
    scale_x, scale_y = resized_w / width, resized_h / height
    padded = Image.new("RGB", (resolution, resolution), (114, 114, 114))
    padded.paste(image.resize((resized_w, resized_h), Image.Resampling.BILINEAR), (left, top))
    geometry = {"left": left, "top": top, "scale_x": scale_x, "scale_y": scale_y}
    if target is not None:
        annotations = []
        for source in target["annotations"]:
            x, y, w, h = source["bbox"]
            box = [x * scale_x + left, y * scale_y + top, w * scale_x, h * scale_y]
            annotations.append(dict(source, bbox=box, area=box[2] * box[3]))
        target = dict(target, annotations=annotations)
    return padded, target, geometry


def restore_tiles(tiles, geometry):
    if geometry is None:
        return tiles
    restored = []
    for tile in tiles:
        x, y, w, h = tile["bbox"]
        restored.append(dict(tile, bbox=[(x - geometry["left"]) / geometry["scale_x"],
            (y - geometry["top"]) / geometry["scale_y"], w / geometry["scale_x"], h / geometry["scale_y"]]))
    return restored



def postprocess_tiles(processor, outputs, sizes: list[tuple[int, int]], classes: dict[str, str], threshold: float):
    """Return raw XYWH tiles in each original, already-oriented image's pixels."""
    results = processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=threshold)
    batches = []
    for result in results:
        tiles = []
        for score, label, box in zip(result["scores"].detach().cpu().tolist(),
                                     result["labels"].detach().cpu().tolist(),
                                     result["boxes"].detach().cpu().tolist()):
            token = classes[str(int(label))]
            x1, y1, x2, y2 = map(float, box)
            tiles.append({"classId": token, "label": token, "bbox": [x1, y1, x2 - x1, y2 - y1],
                          "score": float(score)})
        batches.append(tiles)
    return batches


def verify_model_directory(model_dir: str | Path) -> tuple[Path, dict]:
    path = Path(model_dir)
    manifest = json.loads((path / "model.json").read_text(encoding="utf-8"))
    if manifest.get("architecture") != "rtdetr-v2-r18vd" or manifest.get("format") != "huggingface-pretrained":
        raise ValueError("Expected a packaged Hugging Face RT-DETRv2 R18 model")
    expected = {"config.json", "preprocessor_config.json", "model.safetensors"}
    if set(manifest.get("filesSha256", {})) != expected:
        raise ValueError("Model manifest must hash configuration, processor and safetensors")
    for filename, digest in manifest["filesSha256"].items():
        if file_sha256(model_file(path, filename)) != digest:
            raise ValueError(f"Model file SHA256 mismatch: {filename}")
    validate_labels(manifest)
    if manifest.get("inputMode", "warp") not in {"warp", "letterbox"}:
        raise ValueError("Invalid saved input mode")
    return path, manifest


class RTDETRPredictor:
    def __init__(self, model_dir: str, device: str = "cuda"):
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        path, self.manifest = verify_model_directory(model_dir)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing implicit CPU inference")
        self.processor = RTDetrImageProcessor.from_pretrained(path, local_files_only=True)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(path, local_files_only=True).to(self.device).eval()
        self.classes = self.manifest["classes"]
        if {str(k): v for k, v in self.model.config.id2label.items()} != self.classes:
            raise ValueError("Saved model class mapping differs from model.json")
        expected_size = {"height": self.manifest["resolution"], "width": self.manifest["resolution"]}
        if self.processor.size != expected_size:
            raise ValueError("Saved processor resolution differs from model.json")
        self.version = self.manifest["version"]

    def predict(self, image):
        import torch

        # The HTTP service has already applied EXIF transpose. Do not transform a
        # second time here: returned boxes must describe exactly that PIL image.
        prepared, _, geometry = prepare_input(image.convert("RGB"), resolution=self.manifest["resolution"],
                                             mode=self.manifest.get("inputMode", "warp"))
        inputs = self.processor(images=prepared, return_tensors="pt").to(self.device)
        amp = self.manifest.get("amp", "fp16")
        dtype = torch.bfloat16 if amp == "bf16" else torch.float16
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=dtype, enabled=self.device.type == "cuda" and amp != "none"
        ):
            outputs = self.model(**inputs)
        tiles = postprocess_tiles(self.processor, outputs, [(prepared.height, prepared.width)],
                                  self.classes, self.manifest["threshold"])[0]
        tiles = restore_tiles(tiles, geometry)
        labels = self.manifest.get("labels", {})
        for tile in tiles:
            tile["label"] = labels.get(tile["classId"], tile["classId"])
        return tiles
