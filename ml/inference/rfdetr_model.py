"""RF-DETR adapter. Only immutable manifests and matching checkpoints are accepted."""
import json
from pathlib import Path
from ml.inference.manifest import file_sha256, model_file, validate_labels


def verify_model_directory(model_dir):
    path = Path(model_dir)
    manifest = json.loads((path / "model.json").read_text(encoding="utf-8"))
    validate_labels(manifest)
    if manifest["architecture"] not in {"rfdetr-small", "rfdetr-medium", "rfdetr-large"}:
        raise ValueError("Unsupported architecture")
    checkpoint = model_file(path, manifest["checkpoint"])
    if file_sha256(checkpoint) != manifest["checkpointSha256"]:
        raise ValueError("checkpoint SHA256 mismatch")
    return path, manifest


class RFDETRPredictor:
    def __init__(self, model_dir: str, device="cuda"):
        from rfdetr import RFDETR
        path, self.manifest = verify_model_directory(model_dir)
        checkpoint = model_file(path, self.manifest["checkpoint"])
        self.model = RFDETR.from_checkpoint(str(checkpoint), resolution=self.manifest["resolution"], device=device)
        self.version = self.manifest["version"]
        self.class_names = self.manifest["classes"]
        expected_class = {"rfdetr-small": "RFDETRSmall", "rfdetr-medium": "RFDETRMedium", "rfdetr-large": "RFDETRLarge"}[self.manifest["architecture"]]
        if type(self.model).__name__ != expected_class:
            raise ValueError("Checkpoint architecture differs from manifest")
        if self.model.model_config.num_classes != len(self.class_names):
            raise ValueError("Checkpoint classifier size differs from manifest")
        saved_names = getattr(self.model.model, "class_names", None)
        if saved_names and list(saved_names) != [self.class_names[str(i)] for i in range(len(self.class_names))]:
            raise ValueError("Checkpoint class names differ from manifest")
        self.labels = self.manifest.get("labels", {})

    def predict(self, image):
        import torch
        with torch.inference_mode():
            predictions = self.model.predict(image, threshold=self.manifest["threshold"])
        result = []
        for xyxy, score, class_id in zip(predictions.xyxy, predictions.confidence, predictions.class_id):
            name = self.class_names[str(int(class_id))]
            x1, y1, x2, y2 = map(float, xyxy)
            result.append({"classId": name, "label": self.labels.get(name, name),
                           "score": float(score), "bbox": [x1, y1, x2 - x1, y2 - y1]})
        return result
