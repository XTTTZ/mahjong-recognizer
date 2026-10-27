"""Offline single-GPU YOLO comparison; metadata preflight never starts training."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from unittest.mock import patch

from ml.data.prepare_yolo import verify_prepared
from ml.train.provenance import atomic_json, dataset_contract, sha256

ULTRALYTICS_VERSION = "8.4.189"


def admit_output(output):
    output = Path(output)
    if output.exists():
        allowed = {"status.json", "stdout.log", "stderr.log"}
        if not output.is_dir() or any(not path.is_file() or path.name not in allowed for path in output.iterdir()):
            raise FileExistsError("Use an independent new run; only fresh queue runner logs are allowed")
    else:
        output.mkdir(parents=True)


def enforce_fixed_batch(trainer, expected_batch):
    if trainer.batch_size != expected_batch or trainer.args.batch != expected_batch:
        raise RuntimeError("Ultralytics changed the signed requested batch size")
    # Pinned 8.4.189 initializes this immediately before on_train_epoch_start;
    # its OOM branch re-raises once retries>=3. Keep each independent experiment
    # fixed instead of silently changing batch after part of an epoch.
    trainer._oom_retries = 3


def training_options(args):
    return {"data": str(args.prepared / "data.yaml"), "epochs": args.epochs,
        "imgsz": args.resolution, "batch": args.batch, "nbs": 32,
        "workers": args.workers, "device": 0, "seed": args.seed,
        "optimizer": "AdamW", "lr0": args.lr, "lrf": .05, "weight_decay": .0005,
        "warmup_epochs": 3, "warmup_bias_lr": .0, "cos_lr": True,
        "patience": 15, "amp": "bf16", "deterministic": True,
        "rect": False, "multi_scale": 0., "mosaic": 0., "mixup": 0., "cutmix": 0.,
        "copy_paste": 0., "flipud": 0., "fliplr": 0., "translate": 0., "scale": 0.,
        "degrees": 0., "shear": 0., "perspective": 0., "bgr": 0.,
        "hsv_h": .008 if args.augmentation == "photometric" else 0.,
        "hsv_s": .15 if args.augmentation == "photometric" else 0.,
        "hsv_v": .2 if args.augmentation == "photometric" else 0.,
        "augmentations": [], "plots": False, "val": True, "save": True,
        "save_period": -1, "cache": False, "project": str(args.output), "name": "train",
        "exist_ok": False, "verbose": False, "max_det": 300, "iou": .6,
        "cls_remap": False}


def shared_signature(contract, config, weights):
    root = Path(__file__).resolve().parents[1]
    files = ["train/yolo_train.py", "data/prepare_yolo.py", "train/provenance.py",
             "inference/yolo_model.py", "inference/cascade_model.py", "inference/postprocess.py",
             "inference/manifest.py", "eval/metrics.py", "eval/operating_point.py"]
    runtime = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "ultralytics", "numpy", "Pillow", "opencv-python", "faster-coco-eval"):
        try:
            runtime[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            runtime[name] = None
    return {"data": contract["splits"], "classes": contract["classes"], "config": config,
        "pretrainedSha256": sha256(weights), "code": {name: sha256(root / name) for name in files}, "runtime": runtime}


def checkpoint_receipt(path, signature, classes):
    atomic_json(Path(str(path) + ".receipt.json"), {"checkpointSha256": sha256(path),
        "signatureSha256": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(), "classes": classes})


def verify_receipt(checkpoint, run):
    checkpoint = Path(checkpoint)
    receipt = json.loads(Path(str(checkpoint) + ".receipt.json").read_text(encoding="utf-8"))
    signature_sha = hashlib.sha256(json.dumps(run["resumeSignature"], sort_keys=True).encode()).hexdigest()
    expected = {"0": "tile"} if run.get("generic") else run["classes"]
    if receipt["checkpointSha256"] != sha256(checkpoint) or receipt["signatureSha256"] != signature_sha or receipt["classes"] != expected:
        raise ValueError("Checkpoint receipt differs from the signed training run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--augmentation", choices=("none", "photometric"), default="photometric")
    parser.add_argument("--generic", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.resolution < 320 or args.resolution % 32 or min(args.epochs, args.batch) < 1 or args.workers < 0 or not 0 < args.lr < .1:
        raise ValueError("Invalid YOLO training configuration")
    contract = dataset_contract(args.dataset, verify_images=True)
    if contract["ontology"] != "faces42":
        raise ValueError("YOLO suite requires 42-face data")
    ready = (args.prepared / "prepared.json").is_file()
    if ready:
        verify_prepared(args.prepared, contract, args.generic)
    if not args.weights.is_file():
        raise FileNotFoundError("Cache the official pretrained checkpoint first")
    if args.weights.stem not in {"yolov8s", "yolo26s", "yolo26m"}:
        raise ValueError("Only reviewed pretrained model families are supported")
    options = training_options(args)
    config = {key: value for key, value in options.items() if key not in {"data", "project", "name"}}
    config["oomPolicy"] = "fail_without_changing_batch_then_queue_continues"
    signature = shared_signature(contract, config, args.weights)
    signature["derivativeSha256"] = sha256(args.prepared / "prepared.json") if ready else None
    if args.preflight_only:
        print(json.dumps({"dataContract": contract, "derivativeReady": ready,
            "weightsSha256": signature["pretrainedSha256"], "config": config, "trainingStarted": False}, indent=2))
        return
    if not ready:
        raise FileNotFoundError("Prepare YOLO labels before starting training")
    if (args.output / "run.json").exists():
        raise FileExistsError("Use an independent new run directory")
    if importlib.metadata.version("ultralytics") != ULTRALYTICS_VERSION:
        raise RuntimeError("Use the pinned YOLO environment")
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The prepared experiment requires a CUDA GPU with BF16 support")
    admit_output(args.output)
    os.environ.update({"YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false", "YOLO_CONFIG_DIR": str(args.output / "yolo-config")})
    from ultralytics import YOLO, settings
    settings.update({"sync": False, "wandb": False, "mlflow": False, "clearml": False, "comet": False,
                     "dvc": False, "raytune": False, "tensorboard": False})
    run = {"architecture": args.weights.stem + ("-generic" if args.generic else ""),
        "generic": args.generic, "resolution": args.resolution, "classes": contract["classes"],
        "ontology": contract["ontology"], "resumeSignature": signature,
        "config": config, "startedAt": datetime.now(timezone.utc).isoformat(),
        "annotationSha256": contract["splits"]["train"]["annotationSha256"],
        "validationAnnotationSha256": contract["splits"]["valid"]["annotationSha256"],
        "pretrainedWeights": str(args.weights), "testOpened": False}
    atomic_json(args.output / "run.json", run)
    model = YOLO(str(args.weights), task="detect")
    checkpoint_classes = {"0": "tile"} if args.generic else contract["classes"]
    def saved(trainer):
        for checkpoint in (trainer.last, trainer.best):
            if checkpoint.is_file():
                checkpoint_receipt(checkpoint, signature, checkpoint_classes)
    def metrics(trainer):
        if not trainer.csv.exists():
            return
        rows = list(csv.DictReader(trainer.csv.open(encoding="utf-8")))
        with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "train/loss", "mAP50", "mAP50_95"])
            writer.writeheader()
            for row in rows:
                row = {key.strip(): value for key, value in row.items()}
                losses = [float(value) for key, value in row.items() if key.startswith("train/") and key.endswith("loss")]
                if not all(__import__("math").isfinite(value) for value in losses):
                    raise RuntimeError("Non-finite YOLO loss; terminate this experiment")
                writer.writerow({"epoch": int(row["epoch"])-1, "train/loss": sum(losses),
                    "mAP50": row.get("metrics/mAP50(B)", ""), "mAP50_95": row.get("metrics/mAP50-95(B)", "")})
    model.add_callback("on_model_save", saved)
    model.add_callback("on_fit_epoch_end", metrics)
    model.add_callback("on_train_epoch_start", lambda trainer: enforce_fixed_batch(trainer, args.batch))
    # plots=False removes font use; prevent the dataset checker from fetching fonts anyway.
    with patch("ultralytics.data.utils.check_font", return_value=None):
        model.train(**options)
    for checkpoint in (args.output / "train/weights/last.pt", args.output / "train/weights/best.pt"):
        if checkpoint.is_file():
            checkpoint_receipt(checkpoint, signature, checkpoint_classes)
    run.update(finishedAt=datetime.now(timezone.utc).isoformat(),
        actualBatch=model.trainer.batch_size, actualNominalAccumulate=model.trainer.accumulate,
        actualTrainerArgs=json.loads(json.dumps(vars(model.trainer.args), default=str)),
        checkpointSelection="YOLO best validation AP; downstream hand-metric evaluation is required")
    atomic_json(args.output / "run.json", run)


if __name__ == "__main__":
    main()
