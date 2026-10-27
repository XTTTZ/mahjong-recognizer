"""Grouped truth-crop classifier training. Crop accuracy is diagnostic, never hand accuracy."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

from ml.data.prepare_crops import verify_prepared
from ml.data.prepare_crop_cache_v6 import verify_cache
from ml.train.v6_common import parent_contract, completed_epoch_budget
from ml.train.v6_augment import camera_augment,cached_crop_face
from ml.inference.cascade_model import build_classifier, crop_face, image_tensor
from ml.train.provenance import atomic_json, dataset_contract, sha256
from ml.train.yolo_train import admit_output, checkpoint_receipt, shared_signature


class CropDataset:
    def __init__(self, dataset, crops, split, resolution=224, context=.08, augment=False, rotation="quarter-turn", camera_profile="none"):
        self.root = Path(crops)
        self.records = json.loads((Path(crops) / f"{split}.json").read_text(encoding="utf-8"))["records"]
        self.resolution, self.context, self.augment = resolution, context, augment
        self.rotation = rotation
        self.camera_profile = camera_profile

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image
        from ml.train.rtdetr_train import photometric
        record = self.records[index]
        path = (self.root / record["cacheFile"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Crop image path escapes its signed source")
        with Image.open(path) as source:
            face = cached_crop_face(source, record, self.resolution, self.context, random if self.augment else None)
        if self.augment:
            face = photometric(face)
            if self.rotation == "quarter-turn":
                face = face.rotate(random.choice((0, 90, 180, 270)))
            face = camera_augment(face, random, self.camera_profile)
        return image_tensor(face), int(record["label"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-terminal-receipt", type=Path)
    parser.add_argument("--max-train-minutes", type=float, default=80)
    parser.add_argument("--camera-profile", choices=("none","mild"), default="none")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=(224, 320, 384), default=224)
    parser.add_argument("--context", type=float, default=.08)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--rotation", choices=("none", "quarter-turn"), default="quarter-turn")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=.00005)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.context <= .16 or min(args.epochs, args.batch, args.gradient_accumulation) < 1 or args.workers < 0 or not 0 < args.lr < .1 or args.max_train_minutes <= 0 or not math.isfinite(args.max_train_minutes):
        raise ValueError("Invalid classifier configuration")
    contract = dataset_contract(args.dataset, verify_images=True)
    if contract["ontology"] != "faces42":
        raise ValueError("Cascade requires reviewed 42-face data")
    ready = (args.crops / "prepared.json").is_file() and (args.cache / "prepared.json").is_file()
    if ready:
        verify_prepared(args.crops, contract)
        verify_cache(args.cache, contract, args.crops)
    parent, parent_evidence = parent_contract(args.parent_run,args.parent_checkpoint,contract,args.parent_terminal_receipt,generic=False)
    if parent.get("architecture") != "efficientnet-v2-s":
        raise ValueError("Use a reviewed 42-class EfficientNetV2-S parent")
    config = {"epochs": args.epochs, "resolution": args.resolution, "context": args.context,
        "batch": args.batch, "workers": args.workers, "seed": args.seed, "lr": args.lr,
        "gradientAccumulation": args.gradient_accumulation, "cameraProfile": args.camera_profile,
        "maximumTrainMinutes": args.max_train_minutes, "losslessEnvelopeCache": True,
        "rotation": args.rotation,
        "augmentation": "photometric_and_crop_jitter", "labelSmoothing": .02,
        "classBalance": "inverse_sqrt_capped_at_four", "inputMode": "complete_crop_letterbox_imagenet_normalization"}
    signature = shared_signature(contract, config, args.parent_checkpoint)
    signature["derivativeSha256"] = sha256(args.crops / "prepared.json") if ready else None
    signature["cacheSha256"] = sha256(args.cache / "prepared.json") if ready else None
    signature["parent"] = parent_evidence
    signature["code"].update({"train/cascade_train_v6.py": sha256(__file__),
        "train/v6_common.py": sha256(Path(__file__).resolve().parent/"v6_common.py"),
        "train/v6_augment.py": sha256(Path(__file__).resolve().parent/"v6_augment.py"),
        "data/prepare_crop_cache_v6.py": sha256(Path(__file__).resolve().parents[1]/"data/prepare_crop_cache_v6.py"),
        "data/prepare_crops.py": sha256(Path(__file__).resolve().parents[1] / "data/prepare_crops.py"),
        "train/rtdetr_train.py": sha256(Path(__file__).resolve().parent / "rtdetr_train.py")})
    if args.preflight_only:
        print(json.dumps({"dataContract": contract, "derivativeReady": ready, "config": config,
            "weightsSha256": signature["pretrainedSha256"], "trainingStarted": False}, indent=2))
        return
    if not ready:
        raise FileNotFoundError("Prepare grouped crop indices before training")
    if (args.output / "run.json").exists():
        raise FileExistsError("Use an independent fresh classifier run")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from ml.train.rtdetr_train import seed_worker
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The prepared classifier requires CUDA with BF16")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    admit_output(args.output)
    run = {"architecture": "efficientnet-v2-s", "classes": contract["classes"], "ontology": "faces42",
        "resolution": args.resolution, "config": config, "resumeSignature": signature,
        "startedAt": datetime.now(timezone.utc).isoformat(), "testOpened": False,
        "parentWarmStart": parent_evidence, "plannedEpochs": args.epochs,
        "annotationSha256": contract["splits"]["train"]["annotationSha256"],
        "validationAnnotationSha256": contract["splits"]["valid"]["annotationSha256"],
        "validationScope": "oracle_truth_crops_diagnostic_only"}
    atomic_json(args.output / "run.json", run)
    model = build_classifier()
    saved_parent = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
    expected_signature=hashlib.sha256(json.dumps(parent["resumeSignature"],sort_keys=True).encode()).hexdigest()
    if saved_parent["classes"]!=contract["classes"] or saved_parent["signatureSha256"]!=expected_signature:
        raise ValueError("Parent classifier tensor artifact differs from signed source")
    model.load_state_dict(saved_parent["model"],strict=True)
    model.cuda()
    train = CropDataset(args.dataset, args.cache, "train", args.resolution, args.context, True, args.rotation, args.camera_profile)
    valid = CropDataset(args.dataset, args.cache, "valid", args.resolution, args.context, False)
    counts = Counter(row["label"] for row in train.records)
    if set(counts) != set(range(42)):
        raise ValueError("Classifier training requires examples from all 42 reviewed classes")
    weights = torch.tensor([min(4., (len(train)/(42*counts[i]))**.5) for i in range(42)], device="cuda")
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=.02)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, worker_init_fn=seed_worker, generator=generator)
    valid_loader = DataLoader(valid, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, worker_init_fn=seed_worker)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*.05)
    best, stale = -1., 0
    started=time.monotonic(); completed=0; completion="epoch_limit_reached"
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train/loss", "valid/loss", "oracle_crop_top1", "learningRate"])
        writer.writeheader()
        for epoch in range(args.epochs):
            epoch_started=time.monotonic()
            model.train()
            total_loss, seen = 0., 0
            optimizer.zero_grad(set_to_none=True)
            for step, (images, labels) in enumerate(train_loader):
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = criterion(model(images), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite classifier loss; continue with the next independent queue job")
                remaining = len(train_loader) % args.gradient_accumulation
                denominator = remaining if remaining and step >= len(train_loader)-remaining else args.gradient_accumulation
                (loss/denominator).backward()
                if (step+1) % args.gradient_accumulation == 0 or step+1 == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                total_loss += float(loss.detach())*len(labels)
                seen += len(labels)
            model.eval()
            correct, valid_count, valid_loss = 0, 0, 0.
            with torch.inference_mode():
                for images, labels in valid_loader:
                    images, labels = images.cuda(), labels.cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        outputs = model(images)
                        loss = nn.functional.cross_entropy(outputs, labels)
                    if not torch.isfinite(loss) or not torch.isfinite(outputs).all():
                        raise RuntimeError("Non-finite classifier validation output")
                    correct += int((outputs.argmax(1) == labels).sum())
                    valid_count += len(labels)
                    valid_loss += float(loss)*len(labels)
            accuracy = correct/valid_count
            row = {"epoch": epoch, "train/loss": total_loss/seen, "valid/loss": valid_loss/valid_count,
                "oracle_crop_top1": accuracy, "learningRate": optimizer.param_groups[0]["lr"]}
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
            scheduler.step()
            saved = {"model": model.state_dict(), "classes": contract["classes"], "config": config,
                "signatureSha256": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
                "epoch": epoch, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "pythonRngState": random.getstate(), "torchRngState": torch.get_rng_state(),
                "cudaRngStates": torch.cuda.get_rng_state_all(), "loaderRngState": generator.get_state()}
            temporary = args.output / "last.pt.tmp"
            torch.save(saved, temporary)
            temporary.replace(args.output / "last.pt")
            checkpoint_receipt(args.output / "last.pt", signature, contract["classes"])
            if accuracy > best:
                best, stale = accuracy, 0
                temporary = args.output / "best.pt.tmp"
                torch.save(saved, temporary)
                temporary.replace(args.output / "best.pt")
                checkpoint_receipt(args.output / "best.pt", signature, contract["classes"])
            else:
                stale += 1
            completed=epoch+1
            reason=completed_epoch_budget(completed,args.epochs,time.monotonic()-started,args.max_train_minutes,time.monotonic()-epoch_started)
            atomic_json(args.output/"epoch-completion.json", {"completedEpochs":completed,"checkpointSavedAndValidated":True,"elapsedTrainSeconds":time.monotonic()-started,"stopReason":reason})
            if reason:
                completion=reason
                break
    run.update(finishedAt=datetime.now(timezone.utc).isoformat(), bestOracleCropTop1=best,
        completedEpochs=completed,plannedEpochs=args.epochs,completionReason=completion,convergenceClaim=False,
        elapsedTrainSeconds=time.monotonic()-started,
        note="New independent warm-start; budget_finished is a complete saved epoch, not convergence. Oracle crop accuracy is diagnostic only.")
    atomic_json(args.output / "run.json", run)


if __name__ == "__main__":
    main()
