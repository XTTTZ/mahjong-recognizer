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

from ml.data.prepare_crops import verify_prepared
from ml.inference.cascade_model import build_classifier, crop_face, image_tensor
from ml.train.provenance import atomic_json, dataset_contract, sha256
from ml.train.yolo_train import admit_output, checkpoint_receipt, shared_signature


class CropDataset:
    def __init__(self, dataset, crops, split, resolution=224, context=.08, augment=False, rotation="quarter-turn"):
        self.root = Path(dataset) / split
        self.records = json.loads((Path(crops) / f"{split}.json").read_text(encoding="utf-8"))["records"]
        self.resolution, self.context, self.augment = resolution, context, augment
        self.rotation = rotation

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image
        from ml.train.rtdetr_train import photometric
        record = self.records[index]
        path = (self.root / record["image"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Crop image path escapes its signed source")
        with Image.open(path) as source:
            face = crop_face(source, record["bbox"], self.resolution, self.context, random if self.augment else None)
        if self.augment:
            face = photometric(face)
            if self.rotation == "quarter-turn":
                face = face.rotate(random.choice((0, 90, 180, 270)))
        return image_tensor(face), int(record["label"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=(224, 320, 384), default=224)
    parser.add_argument("--context", type=float, default=.08)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--rotation", choices=("none", "quarter-turn"), default="quarter-turn")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=.0002)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.context <= .3 or min(args.epochs, args.batch, args.gradient_accumulation) < 1 or args.workers < 0 or not 0 < args.lr < .1:
        raise ValueError("Invalid classifier configuration")
    contract = dataset_contract(args.dataset, verify_images=True)
    if contract["ontology"] != "faces42":
        raise ValueError("Cascade requires reviewed 42-face data")
    ready = (args.crops / "prepared.json").is_file()
    if ready:
        verify_prepared(args.crops, contract)
    if not args.weights.is_file() or not sha256(args.weights).startswith("dd5fe13b"):
        raise ValueError("Use the exact official EfficientNetV2-S pretrained checkpoint")
    config = {"epochs": args.epochs, "resolution": args.resolution, "context": args.context,
        "batch": args.batch, "workers": args.workers, "seed": args.seed, "lr": args.lr,
        "gradientAccumulation": args.gradient_accumulation,
        "rotation": args.rotation,
        "augmentation": "photometric_and_crop_jitter", "labelSmoothing": .02,
        "classBalance": "inverse_sqrt_capped_at_four", "inputMode": "complete_crop_letterbox_imagenet_normalization"}
    signature = shared_signature(contract, config, args.weights)
    signature["derivativeSha256"] = sha256(args.crops / "prepared.json") if ready else None
    signature["code"].update({"train/cascade_train.py": sha256(__file__),
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
        "annotationSha256": contract["splits"]["train"]["annotationSha256"],
        "validationAnnotationSha256": contract["splits"]["valid"]["annotationSha256"],
        "validationScope": "oracle_truth_crops_diagnostic_only"}
    atomic_json(args.output / "run.json", run)
    model = build_classifier()
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    state = {key: value for key, value in state.items() if not key.startswith("classifier.1.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {"classifier.1.weight", "classifier.1.bias"} or unexpected:
        raise ValueError("Official classifier weights differ from expected EfficientNetV2-S")
    model.cuda()
    train = CropDataset(args.dataset, args.crops, "train", args.resolution, args.context, True, args.rotation)
    valid = CropDataset(args.dataset, args.crops, "valid", args.resolution, args.context, False)
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
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train/loss", "valid/loss", "oracle_crop_top1", "learningRate"])
        writer.writeheader()
        for epoch in range(args.epochs):
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
                "epoch": epoch, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
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
            if stale >= 10 and epoch >= 14:
                break
    run.update(finishedAt=datetime.now(timezone.utc).isoformat(), bestOracleCropTop1=best,
        note="No complete-hand claim; use cascade_evaluate with detector predictions")
    atomic_json(args.output / "run.json", run)


if __name__ == "__main__":
    main()
