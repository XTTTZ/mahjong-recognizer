"""Lossless train/valid COCO -> YOLO derivative. The sealed test split stays closed."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil

from ml.train.provenance import dataset_contract, sha256, atomic_json


def normalized_box(box, width, height):
    x, y, w, h = box
    return [(x+w/2)/width, (y+h/2)/height, w/width, h/height]


def prepare(dataset, out, generic=False, preflight_only=False):
    dataset, out = Path(dataset).resolve(), Path(out).resolve()
    contract = dataset_contract(dataset, verify_images=True)
    if contract["ontology"] != "faces42":
        raise ValueError("YOLO comparison requires the reviewed 42-face dataset")
    if preflight_only:
        ready = (out / "prepared.json").is_file()
        if ready:
            verify_prepared(out, contract, generic)
        return {"dataContract": contract, "generic": generic, "testOpened": False, "derivativeReady": ready}
    if out.exists():
        raise FileExistsError("Use a fresh derivative directory")
    out.mkdir(parents=True)
    label_digests, image_digests = [], []
    for split in ("train", "valid"):
        images, labels = out / "images" / split, out / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        coco = json.loads((dataset / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        annotations = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations[annotation["image_id"]].append(annotation)
        for image in coco["images"]:
            source = dataset / split / image["file_name"]
            target = images / (str(image["id"]) + source.suffix.lower())
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
            image_digests.append((str(target.relative_to(out)), sha256(target)))
            lines = []
            for annotation in annotations[image["id"]]:
                category = 0 if generic else annotation["category_id"] - 1
                box = normalized_box(annotation["bbox"], image["width"], image["height"])
                lines.append(str(category) + " " + " ".join(f"{value:.12f}" for value in box))
            label = labels / (str(image["id"]) + ".txt")
            label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            label_digests.append((str(label.relative_to(out)), sha256(label)))
    names = ["tile"] if generic else [contract["classes"][str(i)] for i in range(42)]
    # JSON is a YAML subset, avoiding an additional parser during preparation.
    atomic_json(out / "data.yaml", {"path": out.as_posix(), "train": "images/train", "val": "images/valid", "names": names})
    metadata = {"schemaVersion": 1, "kind": "yolo_detection", "generic": generic,
        "dataContract": contract, "classes": contract["classes"], "names": names,
        "labelTreeSha256": hashlib.sha256(json.dumps(sorted(label_digests)).encode()).hexdigest(),
        "images": image_digests,
        "dataYamlSha256": sha256(out / "data.yaml"), "testOpened": False,
        "preparationCodeSha256": sha256(__file__), "derivativeReady": True}
    atomic_json(out / "prepared.json", metadata)
    return metadata


def verify_prepared(prepared, contract, generic=False):
    prepared = Path(prepared)
    metadata = json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))
    if metadata.get("kind") != "yolo_detection" or metadata.get("generic") != generic or metadata.get("dataContract") != contract:
        raise ValueError("YOLO derivative differs from the signed source dataset/task")
    if sha256(prepared / "data.yaml") != metadata["dataYamlSha256"]:
        raise ValueError("YOLO data.yaml changed after preparation")
    digests = [(str(path.relative_to(prepared)), sha256(path)) for path in sorted((prepared / "labels").rglob("*.txt"))]
    if hashlib.sha256(json.dumps(sorted(digests)).encode()).hexdigest() != metadata["labelTreeSha256"]:
        raise ValueError("YOLO labels changed after preparation")
    configuration = json.loads((prepared / "data.yaml").read_text(encoding="utf-8"))
    expected_names = ["tile"] if generic else [contract["classes"][str(i)] for i in range(42)]
    if configuration.get("names") != expected_names or metadata["classes"] != contract["classes"]:
        raise ValueError("YOLO derivative label names differ from the reviewed mapping")
    if Path(configuration["path"]).resolve() != prepared.resolve() or configuration.get("test") or configuration.get("train") != "images/train" or configuration.get("val") != "images/valid":
        raise ValueError("Rebuild derivative on its target host; never point at a different dataset")
    # Derived images are hardlinks when possible; verify bytes again to detect
    # accidental replacement of only the derivative before training.
    for relative, digest in metadata["images"]:
        image = (prepared / relative).resolve()
        if not image.is_relative_to(prepared.resolve()) or sha256(image) != digest:
            raise ValueError("YOLO derivative pixels changed after preparation")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--generic", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.out, args.generic, args.preflight_only), indent=2))


if __name__ == "__main__":
    main()
