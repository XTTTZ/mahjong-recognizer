"""Freeze source-grouped crop indices; crop pixels are read lazily from original images."""
import argparse
import json
from pathlib import Path

from ml.train.provenance import atomic_json, dataset_contract, sha256


def prepare(dataset, out, preflight_only=False):
    dataset, out = Path(dataset).resolve(), Path(out)
    contract = dataset_contract(dataset, verify_images=True)
    if contract["ontology"] != "faces42":
        raise ValueError("Cascade comparison requires the reviewed 42-face dataset")
    if preflight_only:
        ready = (out / "prepared.json").is_file()
        if ready:
            verify_prepared(out, contract)
        return {"dataContract": contract, "testOpened": False, "derivativeReady": ready}
    if out.exists():
        raise FileExistsError("Use a fresh crop-index directory")
    out.mkdir(parents=True)
    indices = {}
    for split in ("train", "valid"):
        coco = json.loads((dataset / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        images = {image["id"]: image for image in coco["images"]}
        records = [{"image": images[a["image_id"]]["file_name"], "imageId": a["image_id"],
            "annotationId": a["id"], "bbox": a["bbox"], "label": a["category_id"]-1,
            "groupId": images[a["image_id"]]["group_id"]} for a in coco["annotations"]]
        atomic_json(out / f"{split}.json", {"records": records})
        indices[split] = {"sha256": sha256(out / f"{split}.json"), "crops": len(records)}
    metadata = {"schemaVersion": 1, "kind": "lazy_classifier_crops", "dataContract": contract,
        "indices": indices, "testOpened": False, "validationScope": "oracle_truth_crops_diagnostic_only",
        "preparationCodeSha256": sha256(__file__), "derivativeReady": True}
    atomic_json(out / "prepared.json", metadata)
    return metadata


def verify_prepared(crops, contract):
    crops = Path(crops)
    metadata = json.loads((crops / "prepared.json").read_text(encoding="utf-8"))
    if metadata.get("kind") != "lazy_classifier_crops" or metadata["dataContract"] != contract:
        raise ValueError("Classifier crop indices differ from source data")
    for split in ("train", "valid"):
        if sha256(crops / f"{split}.json") != metadata["indices"][split]["sha256"]:
            raise ValueError("Classifier crop index changed after preparation")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.out, args.preflight_only), indent=2))


if __name__ == "__main__":
    main()
