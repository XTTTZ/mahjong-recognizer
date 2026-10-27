"""RT-DETRv2 R18 comparison on the exact grouped COCO train/valid splits.

No test split is opened. Run with --help before provisioning a GPU runtime.
Full state checkpoints resume at the next completed epoch boundary.
"""
from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import time

from ml.inference.rtdetr_model import file_sha256, postprocess_tiles, prepare_input, restore_tiles
from ml.train.provenance import dataset_contract
from ml.inference.manifest import class_ontology

SOURCE_MODEL = "PekingU/rtdetr_v2_r18vd"
SOURCE_REVISION = "5650961749fa93567c0d46fc7f43ea4f9e914107"
SOURCE_FILES_SHA256 = {
    "config.json": "ed051ec77cb41c5d9d5e3af21a979b1e890599dfd68434d7636ff82ded4c1527",
    "preprocessor_config.json": "cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1",
    "model.safetensors": "d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115",
}
EXPECTED_TOKENS = {f"{n}{s}" for s in "mps" for n in range(1, 10)} | {
    f"{n}z" for n in range(1, 8)} | {f"{n}h" for n in range(1, 9)} | {"back"}


def write_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def shared_code_hashes(ml_root: Path | None = None):
    root = Path(__file__).resolve().parents[1] if ml_root is None else ml_root
    files = ("eval/metrics.py", "eval/operating_point.py", "inference/postprocess.py",
             "inference/manifest.py", "train/provenance.py")
    return {relative: file_sha256(root / relative) for relative in files}


def runtime_versions(torch_version: str):
    """Called only after GPU preflight, without importing additional frameworks."""
    versions = {"python": platform.python_version(), "torch": str(torch_version)}
    for package in ("transformers", "numpy", "Pillow", "scipy", "faster-coco-eval", "safetensors"):
        versions[package] = importlib.metadata.version(package)
    return versions


def oriented_bbox(bbox, width: int, height: int, orientation: int):
    """Transform a raw COCO XYWH box with the exact PIL EXIF orientation."""
    x, y, w, h = map(float, bbox)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError("Invalid source bbox")
    if x < 0 or y < 0 or x + w > width + 1e-5 or y + h > height + 1e-5:
        raise ValueError("Source annotation is out of bounds; audit it instead of silently changing its truth")

    def point(px, py):
        return {
            1: (px, py), 2: (width - px, py), 3: (width - px, height - py),
            4: (px, height - py), 5: (py, px), 6: (height - py, px),
            7: (height - py, width - px), 8: (py, width - px),
        }[orientation]

    corners = [point(px, py) for px, py in ((x, y), (x + w, y), (x, y + h), (x + w, y + h))]
    output_w, output_h = (height, width) if orientation >= 5 else (width, height)
    xs, ys = zip(*corners)
    x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
    x2, y2 = min(float(output_w), max(xs)), min(float(output_h), max(ys))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Source bbox lies outside the oriented image")
    return [x1, y1, x2 - x1, y2 - y1]


class CocoHands:
    """A structural torch Dataset, with no torch import during metadata checks."""
    def __init__(self, directory: Path, class_ids: dict[int, int] | None = None, limit: int | None = None,
                 augmentation: str = "none"):
        self.directory = directory.resolve()
        self.annotation_path = directory / "_annotations.coco.json"
        if augmentation not in {"none", "photometric"}:
            raise ValueError("Unsupported training augmentation")
        self.augmentation = augmentation
        data = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        categories = sorted(data["categories"], key=lambda c: c["id"])
        tokens = {c["name"] for c in categories}
        if [c["id"] for c in categories] != list(range(1, len(categories) + 1)) or tokens not in (EXPECTED_TOKENS, EXPECTED_TOKENS - {"back"}):
            raise ValueError("Use reviewed 42-face or legacy 43-class COCO data, preserving IDs 1–42/43")
        self.classes = {str(index): category["name"] for index, category in enumerate(categories)}
        self.class_ids = {category["id"]: index for index, category in enumerate(categories)}
        if class_ids is not None and self.class_ids != class_ids:
            raise ValueError("Validation category IDs differ from training IDs")
        self.annotations = defaultdict(list)
        for annotation in data["annotations"]:
            if annotation.get("iscrowd", 0):
                continue
            if annotation["category_id"] not in self.class_ids:
                raise ValueError("Unknown COCO annotation category")
            self.annotations[annotation["image_id"]].append(annotation)
        self.total_records = len(data["images"])
        self.records = data["images"] if limit is None else data["images"][:limit]
        if not self.records:
            raise ValueError("Training and validation sets must not be empty")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image, ImageOps

        record = self.records[index]
        path = (self.directory / record["file_name"]).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("COCO filename escapes the split directory")
        with Image.open(path) as source:
            if source.size != (record["width"], record["height"]):
                raise ValueError(f"COCO dimensions differ from encoded image: {record['file_name']}")
            orientation = int(source.getexif().get(274, 1))
            if orientation not in range(1, 9):
                raise ValueError("Unknown EXIF orientation")
            width, height = source.size
            image = ImageOps.exif_transpose(source).convert("RGB")
        target = []
        for annotation in self.annotations[record["id"]]:
            box = oriented_bbox(annotation["bbox"], width, height, orientation)
            target.append({"id": annotation["id"], "category_id": self.class_ids[annotation["category_id"]],
                           "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
        if self.augmentation == "photometric":
            image = photometric(image)
        return image, {"image_id": record["id"], "annotations": target}, record


def list_collate(batch):
    return batch


def seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_state(generator):
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(), "loader": generator.get_state()}


def restore_rng(state, generator):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def amp_context(torch, amp):
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16) if amp != "none" else nullcontext()


def photometric(image, rng=None):
    """Mild train-only appearance augmentation; tile coordinates never change."""
    from PIL import Image, ImageEnhance, ImageFilter
    rng = random if rng is None else rng
    result = ImageEnhance.Brightness(image).enhance(rng.uniform(.8, 1.2))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(.85, 1.15))
    result = ImageEnhance.Color(result).enhance(rng.uniform(.9, 1.1))
    if rng.random() < .12:
        result = result.filter(ImageFilter.GaussianBlur(rng.uniform(.2, .7)))
    if rng.random() < .2:
        encoded = io.BytesIO()
        result.save(encoded, format="JPEG", quality=rng.randint(75, 95))
        with Image.open(io.BytesIO(encoded.getvalue())) as decoded:
            result = decoded.convert("RGB")
    return result


def coco_metrics(images, annotations, predictions, classes):
    """Standard COCO bbox AP@100 on the exact validation pixels/annotations."""
    if not predictions or not annotations:
        return {"mAP50_95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "maxDetections": 100}
    from faster_coco_eval import COCO, COCOeval_faster
    ground_truth = COCO({"images": images, "annotations": annotations,
                         "categories": [{"id": int(index)+1, "name": token} for index, token in classes.items()]})
    detected = ground_truth.loadRes(predictions)
    evaluator = COCOeval_faster(ground_truth, detected, "bbox", print_function=lambda *args, **kwargs: None)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {"mAP50_95": float(evaluator.stats[0]), "mAP50": float(evaluator.stats[1]),
            "mAP75": float(evaluator.stats[2]), "maxDetections": 100}


def append_metrics(path, epoch, step, loss, chosen, coco, seconds, learning_rate):
    fields = ["epoch", "step", "train/loss", "learning_rate", "elapsed_seconds",
              "val/hand_exact", "val/model_exact", "val/accepted_coverage", "val/mAP_50", "val/mAP_50_95"]
    record = {"epoch": epoch - 1, "step": step, "train/loss": loss,
              "learning_rate": learning_rate, "elapsed_seconds": seconds}
    if chosen:
        record.update({"val/hand_exact": chosen["hand_exact_match"], "val/model_exact": chosen["model_exact"],
                       "val/accepted_coverage": chosen["accepted_coverage"]})
    if coco:
        record.update({"val/mAP_50": coco["mAP50"], "val/mAP_50_95": coco["mAP50_95"]})
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def validate(model, processor, dataset, classes, thresholds, accept_threshold, amp, *, input_mode="warp", resolution=640):
    import torch
    from ml.eval.metrics import Tile, summarize
    from ml.eval.operating_point import score_predictions

    scores = {threshold: [] for threshold in thresholds}
    hand_scores = {threshold: [] for threshold in thresholds}
    scope_scores = {scope: {threshold: [] for threshold in thresholds} for scope in ("raw_image_research", "oracle_roi_auxiliary")}
    predictions = []
    coco_images, coco_truth, coco_predictions = [], [], []
    token_ids = {token: int(index) + 1 for index, token in classes.items()}
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for image, target, record in dataset:
            prepared, _, geometry = prepare_input(image, resolution=resolution, mode=input_mode)
            inputs = processor(images=prepared, return_tensors="pt").to("cuda")
            with amp_context(torch, amp):
                outputs = model(**inputs)
            raw = restore_tiles(postprocess_tiles(processor, outputs, [(prepared.height, prepared.width)], classes, 0.0)[0], geometry)
            coco_images.append({"id": record["id"], "width": image.width, "height": image.height})
            for annotation in target["annotations"]:
                coco_truth.append(dict(annotation, id=len(coco_truth)+1, image_id=record["id"], category_id=annotation["category_id"]+1))
            for tile in raw:
                coco_predictions.append({"image_id": record["id"], "category_id": token_ids[tile["classId"]],
                                         "bbox": tile["bbox"], "score": tile["score"]})
            truth = [Tile(classes[str(a["category_id"])], tuple(a["bbox"])) for a in target["annotations"]]
            for threshold in thresholds:
                score = score_predictions(truth, raw, image.width, image.height, threshold, accept_threshold)
                scores[threshold].append(score)
                scope_scores["oracle_roi_auxiliary" if record.get("oracle_derived") else "raw_image_research"][threshold].append(score)
                if len(truth) >= 10:
                    hand_scores[threshold].append(score)
            predictions.append({"image_id": record["id"], "source_id": record.get("source_id"),
                                "group_id": record.get("group_id"), "predictions": raw, "truth_count": len(truth)})
    reports = {str(t): {"all": summarize(scores[t]), "ten_or_more_tiles": summarize(hand_scores[t]),
                        **{scope: summarize(values[t]) for scope, values in scope_scores.items()}} for t in thresholds}
    selection_group = "ten_or_more_tiles" if any(hand_scores.values()) else "all"
    # Keep rejection-aware accuracy primary; use raw model accuracy only to break
    # early-training ties so different score calibration cannot masquerade as acceptance.
    best = max(thresholds, key=lambda t: (reports[str(t)][selection_group]["hand_exact_match"],
                                         reports[str(t)][selection_group]["model_exact"],
                                         reports[str(t)][selection_group]["accepted_coverage"]))
    return {"split": "validation_only", "externalPhoneBenchmark": False, "selectionGroup": selection_group,
            "validationImages": len(dataset), "partialValidation": len(dataset) != dataset.total_records,
            "bestThreshold": best, "acceptThreshold": accept_threshold, "thresholds": reports,
            "coco": coco_metrics(coco_images, coco_truth, coco_predictions, classes),
            "seconds": time.perf_counter() - started}, predictions


def save_candidate(model, processor, output, metadata, report, epoch, amp):
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    processor.save_pretrained(output)
    files = {filename: file_sha256(output / filename) for filename in ("config.json", "preprocessor_config.json", "model.safetensors")}
    manifest = {"architecture": "rtdetr-v2-r18vd", "format": "huggingface-pretrained",
                "version": f"{metadata['runName']}-epoch-{epoch:03d}", "resolution": metadata["config"]["resolution"],
                "inputMode": metadata["config"]["input_mode"], "ontology": class_ontology(metadata["classes"]),
                "sourceModel": metadata["sourceModel"], "sourceRevision": metadata["sourceRevision"],
                "classes": metadata["classes"], "labels": metadata["labels"], "filesSha256": files,
                "threshold": report["bestThreshold"], "acceptThreshold": report["acceptThreshold"], "amp": amp,
                "dataAnnotationSha256": metadata["annotationSha256"], "validationAnnotationSha256": metadata["validationAnnotationSha256"],
                "codeCommit": metadata["codeCommit"], "productionValidated": False,
                "validationReport": "validation-report.json"}
    report = dict(report, filesSha256=files, annotationSha256=metadata["annotationSha256"],
                  validationAnnotationSha256=metadata["validationAnnotationSha256"], epoch=epoch)
    write_json(output / "model.json", manifest)
    write_json(output / "validation-report.json", report)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-model", default=SOURCE_MODEL)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--source-directory", type=Path, help="Already verified fixed-revision config/processor/safetensors directory")
    parser.add_argument("--class-map", type=Path, default=Path("docs/data/class-map.json"))
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=.05)
    parser.add_argument("--max-grad-norm", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--thresholds", default="0.15,0.25,0.35,0.5,0.65")
    parser.add_argument("--accept-threshold", type=float, default=.7)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=10, help="Do not early-stop before this epoch count")
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--augmentation", choices=["none", "photometric"], default="none")
    parser.add_argument("--input-mode", choices=["warp", "letterbox"], default="warp")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--valid-limit", type=int)
    parser.add_argument("--preflight-only", action="store_true", help="CPU metadata/integrity checks without GPU or output changes")
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--resume", type=Path, help="Trusted last.pt created by this entrypoint; epoch-boundary resume")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "accumulation", "patience", "validation_interval"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.min_epochs < 0:
        parser.error("--min-epochs must be nonnegative")
    if args.workers < 0 or args.resolution < 320 or args.resolution % 32:
        parser.error("Resolution must be >=320 and divisible by32; workers must be >=0")
    if args.source_model != SOURCE_MODEL or not re.fullmatch(r"[a-f0-9]{40}", args.source_revision):
        parser.error("This adapter supports only the pinned PekingU RT-DETRv2 R18 model and a commit revision")
    if not 0 < args.accept_threshold < 1 or not 0 <= args.warmup_ratio < 1:
        parser.error("Invalid acceptance threshold/warmup ratio")
    if min(args.learning_rate, args.backbone_learning_rate, args.max_grad_norm) <= 0 or args.weight_decay < 0:
        parser.error("Learning rates and max grad norm must be positive; weight decay must be nonnegative")
    for name in ("train_limit", "valid_limit"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main():
    args = arguments()
    data_contract = dataset_contract(args.dataset, verify_images=args.verify_images or bool(args.resume) or not args.preflight_only)
    if args.source_directory:
        for filename, digest in SOURCE_FILES_SHA256.items():
            if file_sha256(args.source_directory / filename) != digest:
                raise ValueError(f"Pretrained source integrity mismatch: {filename}")
    if args.preflight_only:
        print(json.dumps({"event": "preflight_passed", "contract": data_contract,
                          "sourceFilesVerified": bool(args.source_directory), "gpuProbed": False,
                          "dataManifestSha256": file_sha256(args.data_manifest) if args.data_manifest else None,
                          "outputWritten": False}, ensure_ascii=False, indent=2))
        return
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import RTDetrImageProcessor, RTDetrV2Config, RTDetrV2ForObjectDetection
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this command does not silently train on CPU")
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by this GPU/runtime")
    thresholds = sorted({float(v) for v in args.thresholds.split(",")})
    if not thresholds or not all(0 < t < 1 for t in thresholds):
        raise ValueError("Validation confidence thresholds must be in (0,1)")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    train = CocoHands(args.dataset / "train", limit=args.train_limit, augmentation=args.augmentation)
    valid = CocoHands(args.dataset / "valid", train.class_ids, args.valid_limit)
    if train.classes != valid.classes:
        raise ValueError("Training and validation label tokens differ")
    groups_train = {r["group_id"] for r in train.records if r.get("group_id") is not None}
    groups_valid = {r["group_id"] for r in valid.records if r.get("group_id") is not None}
    if groups_train & groups_valid:
        raise ValueError("Grouped train/validation split leakage detected")
    classes = train.classes
    label_data = json.loads(args.class_map.read_text(encoding="utf-8"))
    labels = {c["token"]: c["label_zh"] for c in label_data["classes"] if c["token"] in classes.values()}
    if set(labels) != set(classes.values()):
        raise ValueError("Class map must cover the actual reviewed category tokens")
    annotation_hash = file_sha256(train.annotation_path)
    validation_hash = file_sha256(valid.annotation_path)
    training_config = {k: v for k, v in vars(args).items() if k not in {"resume", "output", "dataset", "class_map", "data_manifest", "source_directory", "preflight_only", "verify_images"}}
    signature = {"config": training_config, "dataContract": data_contract["splits"], "annotationSha256": annotation_hash, "validationAnnotationSha256": validation_hash,
                 "classMapSha256": file_sha256(args.class_map), "classes": classes,
                 "trainingCodeSha256": file_sha256(Path(__file__)),
                 "inferenceCodeSha256": file_sha256(Path(__file__).resolve().parents[1] / "inference" / "rtdetr_model.py"),
                 "sharedCodeSha256": shared_code_hashes(), "runtimeVersions": runtime_versions(torch.__version__),
                 "sourceFilesSha256": SOURCE_FILES_SHA256,
                 "dataManifestSha256": file_sha256(args.data_manifest) if args.data_manifest else None}
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "run.json").exists() and not args.resume:
        raise FileExistsError("Existing run; use a new output or explicit --resume")
    state = None
    if args.resume:
        # Full optimizer/RNG state is pickle-based: accept only our trusted local
        # checkpoint, never a downloaded arbitrary .pt file supplied by a user.
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("schemaVersion") != 1 or state["signature"] != signature:
            raise ValueError("Resume checkpoint does not match the exact dataset/class map/training configuration")
        processor = RTDetrImageProcessor.from_dict(state["processor"])
        model = RTDetrV2ForObjectDetection(RTDetrV2Config.from_dict(state["modelConfig"]))
        model.load_state_dict(state["model"], strict=True)
        if args.resume.resolve().parent != args.output.resolve() and not (args.output / "best").exists():
            previous_best = args.resume.resolve().parent / "best"
            if not previous_best.is_dir():
                raise ValueError("Resume into a new output needs the previous best/ candidate beside last.pt")
            shutil.copytree(previous_best, args.output / "best")
    else:
        if args.source_revision != SOURCE_REVISION:
            raise ValueError("An alternate source revision requires independently updating the reviewed file hashes")
        if args.source_directory:
            source_directory = args.source_directory
        else:
            from huggingface_hub import snapshot_download
            source_directory = Path(snapshot_download(args.source_model, revision=args.source_revision,
                allow_patterns=list(SOURCE_FILES_SHA256)))
        for filename, digest in SOURCE_FILES_SHA256.items():
            if file_sha256(source_directory / filename) != digest:
                raise ValueError(f"Pretrained source integrity mismatch: {filename}")
        processor = RTDetrImageProcessor.from_pretrained(source_directory, local_files_only=True,
            size={"height": args.resolution, "width": args.resolution}, do_pad=False)
        model = RTDetrV2ForObjectDetection.from_pretrained(
            source_directory, local_files_only=True, num_labels=len(classes),
            id2label={int(k): v for k, v in classes.items()}, label2id={v: int(k) for k, v in classes.items()},
            ignore_mismatched_sizes=True, trust_remote_code=False)
    # Dynamic anchors are necessary if evaluating a separately fine-tuned input
    # size. The pinned source already has both static dimensions disabled.
    model.config.anchor_image_size = None
    model.config.eval_size = None
    model.to("cuda")
    groups = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            lr = args.backbone_learning_rate if "backbone" in name else args.learning_rate
            decay = 0.0 if parameter.ndim <= 1 or name.endswith(".bias") else args.weight_decay
            groups[(lr, decay)].append(parameter)
    optimizer = torch.optim.AdamW([{"params": values, "lr": lr, "weight_decay": decay} for (lr, decay), values in groups.items()])
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=list_collate, worker_init_fn=seed_worker, generator=generator,
                        persistent_workers=False, drop_last=False)
    updates_per_epoch = math.ceil(len(loader) / args.accumulation)
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = max(1, round(total_steps * args.warmup_ratio)) if args.warmup_ratio else 0

    def schedule(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return .5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    start_epoch = 0
    best_score = (-1.0, -1.0, -1.0, -1.0)
    stale = 0
    global_step = 0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"], generator)
        start_epoch = state["epoch"]
        best_score = tuple(state["bestScore"])
        stale = state["stale"]
        global_step = state["globalStep"]
        if stale >= args.patience and start_epoch >= args.min_epochs:
            raise RuntimeError("This checkpoint already early-stopped; start a new explicit experiment instead of resuming it")
    try:
        code_commit = os.environ.get("MAHJONG_CODE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        code_commit = None
    metadata = {"schemaVersion": 1, "runName": args.output.name, "startedAt": datetime.now(timezone.utc).isoformat(),
                "architecture": "rtdetr-v2-r18vd", "ontology": class_ontology(classes), "sourceModel": args.source_model, "sourceRevision": args.source_revision,
                **signature, "labels": labels, "dataset": str(args.dataset.resolve()), "codeCommit": code_commit,
                "environment": {**signature["runtimeVersions"], "gpu": torch.cuda.get_device_name(0)},
                "augmentation": {"horizontalFlip": False, "verticalFlip": False, "crop": False, "photometric": args.augmentation,
                    "resize": args.input_mode + "; same prepare_input and saved processor during train/valid/inference",
                    "EXIF": "transpose image and all raw COCO bbox corners together"},
                "productionValidated": False, "resumeFrom": str(args.resume) if args.resume else None}
    write_json(args.output / "run.json", metadata)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, batch in enumerate(loader):
            images, annotations, _ = zip(*batch)
            prepared = [prepare_input(image, target,
                                     resolution=args.resolution, mode=args.input_mode) for image, target in zip(images, annotations)]
            inputs = processor(images=[item[0] for item in prepared], annotations=[item[1] for item in prepared], return_tensors="pt").to("cuda")
            # BatchFeature.to does not move tensors nested inside the list of
            # target dicts in all supported Transformers versions.
            inputs["labels"] = [{key: value.to("cuda") if torch.is_tensor(value) else value
                                 for key, value in target.items()} for target in inputs["labels"]]
            span = min(args.accumulation, len(loader) - (step // args.accumulation) * args.accumulation)
            with amp_context(torch, args.amp):
                loss = model(**inputs).loss
            if loss is None or not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite training loss; do not continue an invalid run")
            total_loss += float(loss.detach().item())
            scaler.scale(loss / span).backward()
            if (step + 1) % args.accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    global_step += 1
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 20 == 0 or step + 1 == len(loader):
                print(json.dumps({"event": "train", "epoch": epoch + 1, "batch": step + 1,
                                  "batches": len(loader), "meanLoss": total_loss / (step + 1), "globalStep": global_step}), flush=True)
        should_validate = (epoch + 1) == 1 or (epoch + 1) % args.validation_interval == 0 or (epoch + 1) == args.epochs
        if should_validate:
            report, predictions = validate(model, processor, valid, classes, thresholds, args.accept_threshold, args.amp,
                                           input_mode=args.input_mode, resolution=args.resolution)
            chosen = report["thresholds"][str(report["bestThreshold"])][report["selectionGroup"]]
            candidate_score = (chosen["hand_exact_match"], chosen["model_exact"], chosen["accepted_coverage"], report["coco"]["mAP50_95"])
        if should_validate:
            report.update(epoch=epoch + 1, meanTrainLoss=total_loss / len(loader), architecture=metadata["architecture"],
                          sourceModel=args.source_model, sourceRevision=args.source_revision,
                          annotationSha256=annotation_hash, validationAnnotationSha256=validation_hash)
            write_json(args.output / f"validation-{epoch + 1:03d}.json", report)
            if candidate_score > best_score:
                best_score, stale = candidate_score, 0
                save_candidate(model, processor, args.output / "best", metadata, report, epoch + 1, args.amp)
                with (args.output / "best" / "validation-predictions.jsonl").open("w", encoding="utf-8") as output:
                    for prediction in predictions:
                        output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            else:
                stale += 1
        checkpoint = {"schemaVersion": 1, "signature": signature, "epoch": epoch + 1,
                      "model": model.state_dict(), "modelConfig": model.config.to_dict(),
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "scaler": scaler.state_dict(), "rng": rng_state(generator), "processor": processor.to_dict(),
                      "globalStep": global_step, "bestScore": best_score, "stale": stale}
        temporary = args.output / "last.pt.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, args.output / "last.pt")
        append_metrics(args.output / "metrics.csv", epoch + 1, global_step, total_loss / len(loader),
                       chosen if should_validate else None, report.get("coco") if should_validate else None,
                       time.perf_counter() - started, optimizer.param_groups[0]["lr"])
        print(json.dumps({"event": "validated" if should_validate else "epoch_finished", "epoch": epoch + 1,
                          "meanLoss": total_loss / len(loader),
                          "handExact": chosen["hand_exact_match"] if should_validate else None,
                          "mAP50_95": report["coco"]["mAP50_95"] if should_validate else None,
                          "staleValidationRounds": stale}), flush=True)
        if should_validate and stale >= args.patience and epoch + 1 >= args.min_epochs:
            break
    metadata.update(finishedAt=datetime.now(timezone.utc).isoformat(), elapsedSeconds=time.perf_counter() - started,
                    peakCudaBytes=torch.cuda.max_memory_allocated(), completedEpochs=epoch + 1 if start_epoch < args.epochs else start_epoch)
    write_json(args.output / "run.json", metadata)
    print(json.dumps({"event": "train_finished", "output": str(args.output), "completedEpochs": metadata["completedEpochs"]}), flush=True)


if __name__ == "__main__":
    main()
