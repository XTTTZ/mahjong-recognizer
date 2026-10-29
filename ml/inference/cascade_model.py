"""Detect front faces, then classify the actual detector crops as a batch."""
import json
import math
from pathlib import Path

from ml.inference.manifest import file_sha256, model_file, validate_labels


def crop_face(image, box, resolution=224, context=.08, rng=None):
    """Keep the complete glyph, aspect ratio and context; never center-crop it away."""
    from PIL import Image
    x, y, width, height = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, width, height)) or min(width, height) <= 0:
        raise ValueError("Invalid face box")
    if rng is not None:
        x += rng.uniform(-.03, .03)*width
        y += rng.uniform(-.03, .03)*height
        width *= rng.uniform(.94, 1.06)
        height *= rng.uniform(.94, 1.06)
    left, top = max(0, math.floor(x-width*context)), max(0, math.floor(y-height*context))
    right, bottom = min(image.width, math.ceil(x+width*(1+context))), min(image.height, math.ceil(y+height*(1+context)))
    if right <= left or bottom <= top:
        raise ValueError("Face box falls outside the image")
    face = image.crop((left, top, right, bottom)).convert("RGB")
    scale = min(resolution/face.width, resolution/face.height)
    face = face.resize((max(1, round(face.width*scale)), max(1, round(face.height*scale))), Image.Resampling.BILINEAR)
    result = Image.new("RGB", (resolution, resolution), (114, 114, 114))
    result.paste(face, ((resolution-face.width)//2, (resolution-face.height)//2))
    return result


def image_tensor(image):
    import torch
    import numpy as np
    array = np.asarray(image, dtype="float32").copy()/255.
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    return (tensor-mean)/std


def build_classifier():
    from torch import nn
    from torchvision.models import efficientnet_v2_s
    model = efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 42)
    return model


def combine_tiles(detections, probabilities, classes):
    if len(detections) != len(probabilities):
        raise ValueError("Classifier output does not cover all detected faces")
    tiles = []
    for detection, values in zip(detections, probabilities):
        if len(values) != len(classes) or not all(math.isfinite(p) and 0 <= p <= 1 for p in values):
            raise ValueError("Invalid classifier probability vector")
        label = max(range(len(values)), key=values.__getitem__)
        token = classes[str(label)]
        # Conservative conjunction. This remains an uncalibrated selection score,
        # and its operating point is selected only on image-level validation.
        score = float(detection["score"]) * float(values[label])
        tiles.append(dict(detection, classId=token, label=token, score=score,
                          detectorScore=detection["score"], classifierScore=float(values[label])))
    return tiles


class RawCascade:
    def __init__(self, detector, classifier, classes, resolution=224, context=.08, device="cuda"):
        import torch
        self.detector, self.classes, self.device = detector, classes, device
        self.resolution, self.context = resolution, context
        self.classifier = build_classifier()
        saved = torch.load(classifier, map_location="cpu", weights_only=True)
        if saved["classes"] != classes or saved["config"]["resolution"] != resolution or saved["config"]["context"] != context:
            raise ValueError("Classifier checkpoint geometry/classes differ from the signed run")
        self.classifier.load_state_dict(saved["model"])
        self.classifier.to(device).eval()
        self.fast_preprocess = False
        self.runtime_backend = 'torch-bf16'

    def enable_fast_preprocess(self):
        import torch
        # Constants are allocated once and restored with GPU memory snapshots.
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]
        self.fast_preprocess = True

    def crop_batch(self, image, detections):
        import numpy as np
        import torch
        if not self.fast_preprocess:
            return torch.stack([image_tensor(crop_face(image, d['bbox'], self.resolution, self.context))
                                for d in detections]).to(self.device)
        # Preserve the exact PIL bilinear crop/pad geometry used in training.
        # Transfer one contiguous uint8 batch instead of many normalized float
        # tiles, then normalize the full batch on the GPU in float32.
        pixels = np.stack([np.asarray(crop_face(image, d['bbox'], self.resolution, self.context))
                           for d in detections])
        batch = torch.from_numpy(pixels).to(self.device).permute(0, 3, 1, 2).float().div_(255.)
        return batch.sub_(self.mean).div_(self.std).contiguous()

    def warmup(self):
        import torch
        from PIL import Image
        self.detector.predict(Image.new('RGB', (960, 640), (114, 114, 114)))
        for count in (1, 14, 32):
            batch = torch.zeros(count, 3, self.resolution, self.resolution, device=self.device)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16,
                    enabled=self.device == 'cuda' and self.runtime_backend != 'tensorrt-fp16'):
                self.classifier(batch)
        if self.device == 'cuda': torch.cuda.synchronize()

    def predict(self, image):
        import torch
        detections = self.detector.predict(image)
        if len(detections) > 200:
            raise ValueError("Detector exceeded the API safety limit")
        if not detections:
            return []
        outputs = []
        # Bound crop batches for large scenes; no detections are silently dropped.
        for start in range(0, len(detections), 32):
            batch = self.crop_batch(image, detections[start:start+32])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                    enabled=self.device == "cuda" and self.runtime_backend != 'tensorrt-fp16'):
                values = self.classifier(batch).float().softmax(-1).cpu().tolist()
            outputs.extend(values)
        return combine_tiles(detections, outputs, self.classes)


def verify_model_directory(model_dir):
    directory = Path(model_dir)
    manifest = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    if manifest.get("architecture") != "yolo-efficientnet-v2-s-cascade" or manifest.get("format") != "two-stage-pt":
        raise ValueError("Unexpected cascade model format")
    if set(manifest.get("filesSha256", {})) != {"detector.pt", "classifier.pt"}:
        raise ValueError("Cascade manifest must bind both stages")
    for filename, digest in manifest["filesSha256"].items():
        if file_sha256(model_file(directory, filename)) != digest:
            raise ValueError("Cascade stage SHA mismatch")
    validate_labels(manifest)
    if manifest.get("classifierResolution") not in (224, 320, 384) or not 0 <= manifest.get("cropContext", -1) <= .3:
        raise ValueError("Unexpected cascade crop preprocessing")
    return directory, manifest


class CascadePredictor(RawCascade):
    def __init__(self, model_dir, device="cuda"):
        from ml.inference.yolo_model import RawYOLO
        directory, self.manifest = verify_model_directory(model_dir)
        detector = RawYOLO(directory / "detector.pt", self.manifest["classes"], self.manifest["resolution"],
            device, self.manifest.get("detectorThreshold", .05), generic=True, iou=self.manifest.get("nmsIou", .6))
        super().__init__(detector, directory / "classifier.pt", self.manifest["classes"],
            self.manifest["classifierResolution"], self.manifest["cropContext"], device)
        self.version = self.manifest["version"]

    def predict(self, image):
        return [tile for tile in super().predict(image) if tile["score"] > self.manifest["threshold"]]
