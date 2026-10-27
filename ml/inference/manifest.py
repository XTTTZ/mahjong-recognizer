"""Validate packaged metadata without importing a GPU framework."""
import hashlib
import math
from pathlib import Path

FACE_TOKENS = {f"{n}{s}" for s in "mps" for n in range(1, 10)} | {
    f"{n}z" for n in range(1, 8)} | {f"{n}h" for n in range(1, 9)}
TOKENS = FACE_TOKENS | {"back"}


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def class_ontology(classes):
    count = len(classes)
    expected = FACE_TOKENS if count == 42 else TOKENS if count == 43 else set()
    if not expected or set(classes) != {str(i) for i in range(count)} or set(classes.values()) != expected:
        raise ValueError("Expected contiguous mapping for 42 tile faces or legacy 43 classes including back")
    return "faces42" if count == 42 else "legacy43_with_back"


def validate_labels(manifest):
    ontology = class_ontology(manifest["classes"])
    if manifest.get("ontology", ontology) != ontology:
        raise ValueError("Model ontology differs from class mapping")
    for key in ("threshold", "acceptThreshold"):
        value = manifest[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("Invalid saved confidence thresholds")
    if not isinstance(manifest.get("version"), str) or not manifest["version"].strip():
        raise ValueError("Model needs an immutable version name")
    resolution = manifest.get("resolution")
    if type(resolution) is not int or resolution < 320 or resolution % 32:
        raise ValueError("Invalid saved resolution")


def model_file(directory, filename):
    path = Path(directory).resolve()
    if not isinstance(filename, str) or not filename or Path(filename).name != filename or "\\" in filename:
        raise ValueError("Model file must be a model-local filename")
    file = (path / filename).resolve()
    if file.parent != path or not file.is_file():
        raise ValueError("Model file missing or points outside the model directory")
    return file
