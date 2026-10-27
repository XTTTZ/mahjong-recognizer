"""CPU-only training contracts. Never opens the sealed test split."""
import hashlib
import json
import math
from pathlib import Path

from ml.inference.manifest import class_ontology


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dataset_contract(dataset, verify_images=False):
    dataset = Path(dataset).resolve()
    result, groups, names = {}, {}, None
    image_digests = {}
    for split in ("train", "valid"):
        directory = dataset / split
        annotation_path = directory / "_annotations.coco.json"
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = sorted(data["categories"], key=lambda c: c["id"])
        if [c["id"] for c in categories] != list(range(1, len(categories)+1)):
            raise ValueError("COCO categories must preserve reviewed contiguous IDs 1..42 or legacy 1..43")
        classes = {str(i): c["name"] for i, c in enumerate(categories)}
        ontology = class_ontology(classes)
        if names is not None and names != classes:
            raise ValueError("Train/valid label mappings differ")
        names = classes
        images = {r["id"]: r for r in data["images"]}
        if not images or len(images) != len(data["images"]):
            raise ValueError("Each training/validation split needs nonempty, unique image IDs")
        groups[split] = {r["group_id"] for r in images.values() if r.get("group_id")}
        if any(not r.get("group_id") for r in images.values()):
            raise ValueError("Missing source grouping; use the curated materialized data")
        seen_paths = set()
        pixel_hashes = []
        for record in images.values():
            path = (directory / record["file_name"]).resolve()
            if not path.is_relative_to(directory) or path in seen_paths:
                raise ValueError("Unsafe or duplicate image filename")
            seen_paths.add(path)
            if min(record["width"], record["height"]) <= 0:
                raise ValueError("Invalid image dimensions")
            if verify_images:
                from PIL import Image
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    if image.size != (record["width"], record["height"]) or image.getexif().get(274, 1) != 1:
                        raise ValueError(f"Normalize dimensions/EXIF in a reviewed derivative first: {path.name}")
                    image.load()
                digest = sha256(path)
                if record.get("sha256") and record["sha256"] != digest:
                    raise ValueError(f"Image SHA256 differs from curated manifest: {path.name}")
                pixel_hashes.append((str(record["id"]), digest))
        annotation_ids = set()
        for a in data["annotations"]:
            if a["id"] in annotation_ids or a["image_id"] not in images or a["category_id"] not in range(1, len(classes)+1):
                raise ValueError("Invalid or duplicated annotation reference")
            annotation_ids.add(a["id"])
            image = images[a["image_id"]]
            x, y, w, h = a["bbox"]
            if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0 or min(x, y) < 0 or x+w > image["width"]+1e-5 or y+h > image["height"]+1e-5:
                raise ValueError("Out-of-bounds or non-finite truth box; quarantine rather than silently clip")
            if a.get("iscrowd", 0):
                raise ValueError("Crowd annotations are not a complete-hand ground truth")
        result[split] = {"annotationSha256": sha256(annotation_path), "images": len(images),
                         "annotations": len(data["annotations"]), "groups": len(groups[split])}
        if verify_images:
            image_digests[split] = {value for _, value in pixel_hashes}
            result[split]["imageTreeSha256"] = hashlib.sha256(json.dumps(sorted(pixel_hashes)).encode()).hexdigest()
    if groups["train"] & groups["valid"]:
        raise ValueError("Train/validation group leakage detected")
    if verify_images and image_digests["train"] & image_digests["valid"]:
        raise ValueError("Train/validation exact image byte leakage detected even with different group IDs")
    return {"classes": names, "ontology": ontology, "splits": result, "testOpened": False, "imagesVerified": verify_images}


def check_resume(previous, signature, resume, output):
    if previous.get("resumeSignature") != signature:
        raise ValueError("Resume requires identical data, validation, code and training settings; start a new run after curation")
    if Path(resume).resolve().parent != Path(output).resolve():
        raise ValueError("Resume only from a checkpoint in its original run directory")


def check_evaluation_data(run, contract):
    if run.get("resumeSignature", {}).get("data") != contract["splits"]:
        raise ValueError("Evaluation pixels or split differ from signed run; legacy runs remain historical research only")
    if run["classes"] != contract["classes"] or run["annotationSha256"] != contract["splits"]["train"]["annotationSha256"]:
        raise ValueError("Evaluation data/classes differ from the training run")
    if run.get("validationAnnotationSha256") != contract["splits"]["valid"]["annotationSha256"]:
        raise ValueError("Validation changed since this training run")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
