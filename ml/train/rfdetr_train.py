"""Versioned RF-DETR training; CPU preflight never opens the sealed test split."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from ml.train.provenance import atomic_json, check_resume, dataset_contract, sha256
from ml.inference.manifest import class_ontology


def checkpoint_contract(checkpoint, signature, require_training_state=True):
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    notes = checkpoint.get("args", {}).get("notes")
    if not isinstance(notes, dict) or notes.get("mahjongResumeSignature") != digest:
        raise ValueError("Checkpoint belongs to another or legacy run; cannot resume on changed data/code")
    if require_training_state and (not checkpoint.get("optimizer_states") or not checkpoint.get("lr_schedulers") or "epoch" not in checkpoint):
        raise ValueError("Resume needs a complete epoch checkpoint with optimizer and scheduler state")


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2)
    parser.add_argument("--scheduler", choices=["step", "cosine"], default="cosine")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--augmentation", choices=["none", "photometric"], default="none")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--preflight-only", action="store_true", help="Validate data metadata without torch, GPU or output changes")
    parser.add_argument("--verify-images", action="store_true", help="Also decode/check train+valid images; required before a new RF run")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation, args.patience) <= 0 or args.workers < 0:
        parser.error("epochs, batch and accumulation must be positive; workers must be nonnegative")
    if args.resolution < 320 or args.resolution % 64:
        parser.error("RF resolution must be >=320 and divisible by64")
    if min(args.learning_rate, args.encoder_learning_rate) <= 0 or args.weight_decay < 0:
        parser.error("Invalid learning rates/weight decay")
    if args.warmup_epochs < 0 or min(args.eval_interval, args.checkpoint_interval) < 1:
        parser.error("Invalid warmup or checkpoint/validation interval")
    contract = dataset_contract(args.dataset, verify_images=args.verify_images or bool(args.resume) or not args.preflight_only)
    classes = contract["classes"]
    augmentations = {} if args.augmentation == "none" else {
        "RandomBrightnessContrast": {"brightness_limit": 0.18, "contrast_limit": 0.18, "p": 0.5},
        "HueSaturationValue": {"hue_shift_limit": 3, "sat_shift_limit": 12, "val_shift_limit": 8, "p": 0.25},
        "GaussianBlur": {"blur_limit": [3, 3], "sigma_limit": [0.1, 0.65], "p": 0.15},
        "ImageCompression": {"quality_range": [65, 95], "p": 0.2},
    }
    config = {
        "dataset_dir": str(args.dataset.resolve()), "output_dir": str(args.output.resolve()),
        "epochs": args.epochs, "batch_size": args.batch_size, "grad_accum_steps": args.accumulation,
        "num_workers": args.workers, "seed": args.seed, "amp_dtype": "fp16",
        "aug_config": augmentations, "augmentation_backend": "torchvision" if not augmentations else "albumentations", "scale_jitter": False, "multi_scale": False, "expanded_scales": False,
        "checkpoint_interval": args.checkpoint_interval, "early_stopping": not args.no_early_stopping,
        "lr": args.learning_rate, "lr_encoder": args.encoder_learning_rate, "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs, "lr_scheduler": args.scheduler,
        "lr_scheduler_kwargs": {"min_factor": 0.05}, "eval_interval": args.eval_interval, "early_stopping_patience": args.patience,
        "tensorboard": False, "wandb": False, "run_test": False, "device": "cuda",
        "class_names": list(classes.values()), "progress_bar": None,
    }
    signature = {"architecture": "rfdetr-" + args.architecture, "resolution": args.resolution,
        "config": {k: v for k, v in config.items() if k not in {"dataset_dir", "output_dir"}},
        "classes": classes, "data": contract["splits"],
        "dataManifestSha256": sha256(args.data_manifest) if args.data_manifest else None,
        "trainingEntrySha256": sha256(__file__), "contractCodeSha256": sha256(Path(__file__).with_name("provenance.py")),
        "sharedCodeSha256": {name: sha256(Path(__file__).resolve().parents[1] / name) for name in
            ("eval/metrics.py", "eval/operating_point.py", "inference/postprocess.py", "inference/manifest.py")},
        "runtimePackages": {name: package_version(name) for name in
            ("torch", "torchvision", "rfdetr", "numpy", "pillow", "scipy", "vernier", "albumentations")}}
    metadata_path = args.output / "run.json"
    if metadata_path.exists() and not args.resume:
        raise RuntimeError("Run already exists; use a fresh output or explicit --resume")
    previous = None
    if args.resume:
        if not metadata_path.exists():
            raise ValueError("Resume requires original run.json beside the checkpoint")
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        check_resume(previous, signature, args.resume, args.output)
    if args.preflight_only:
        print(json.dumps({"event": "preflight_passed", "contract": contract, "resumeSignature": signature,
                          "gpuProbed": False, "outputWritten": False}, ensure_ascii=False, indent=2))
        return
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    import torch
    from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to silently run full training on CPU")
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "rfdetr": importlib.metadata.version("rfdetr"), "albumentations": importlib.metadata.version("albumentations") if args.augmentation != "none" else None, "gpu": torch.cuda.get_device_name(0)}
    if previous and previous.get("environment") != environment:
        raise ValueError("Resume environment differs; restore the original verified environment")
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    config["notes"] = {"mahjongResumeSignature": digest}
    if args.resume:
        checkpoint_contract(torch.load(args.resume, map_location="cpu", weights_only=True), signature)
        config["resume"] = str(args.resume.resolve())
    try:
        commit = os.environ.get("MAHJONG_CODE_COMMIT") or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    metadata = previous or {
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "architecture": signature["architecture"], "resolution": args.resolution,
        "classes": classes, "cocoCategoryIds": {str(i): i+1 for i in range(len(classes))}, "ontology": class_ontology(classes),
        "annotationSha256": contract["splits"]["train"]["annotationSha256"],
        "validationAnnotationSha256": contract["splits"]["valid"]["annotationSha256"],
        "config": config, "codeCommit": commit, "resumeSignature": signature,
        "trainingEntrySha256": signature["trainingEntrySha256"], "environment": environment,
        "phase": "research_baseline", "productionValidated": False,
        "preprocessing": "RF-DETR native fixed square resize; no flips, crop or scale jitter",
        "augmentation": args.augmentation,
        "resumeSemantics": "epoch/optimizer/scheduler restore; RF library reseeds fit; RNG-equivalent uninterrupted trajectory not verified",
        "note": "mAP checkpoints are candidates; select on validation complete-hand accuracy before final test.",
    }
    if args.resume:
        metadata.setdefault("resumeHistory", []).append({"checkpoint": args.resume.name,
            "sha256": sha256(args.resume), "time": datetime.now(timezone.utc).isoformat()})
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(metadata_path, metadata)
    ctor = {"small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}[args.architecture]
    model = ctor(resolution=args.resolution)
    started = time.perf_counter()
    model.train(**config)
    metadata.update(finishedAt=datetime.now(timezone.utc).isoformat(), elapsedSeconds=round(time.perf_counter() - started, 2),
                    peakCudaBytes=torch.cuda.max_memory_allocated())
    atomic_json(metadata_path, metadata)
    print(json.dumps({"event": "train_finished", "output": str(args.output), "seconds": metadata["elapsedSeconds"]}))


if __name__ == "__main__":
    main()
