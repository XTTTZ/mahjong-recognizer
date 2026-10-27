"""Write separate reproducible RF/RT experiment plans without launching jobs.

Plans are intentionally not job_runner specs: read their preflight commands and
complete the post-boot GPU smoke checks before creating a launchable job.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath

from infra.windows.offline_bundle import digest


def plans(root: Path, metadata: Path, manifest: Path, code_root: str, data_root: str) -> dict[str, dict]:
    data = json.loads(metadata.read_text(encoding="utf-8"))
    if digest(manifest) != data["tracked_split_sha256"]:
        raise ValueError("Dataset metadata and selection manifest differ")
    revision = max(int(data.get(k, 0)) for k in ("audit_revision", "selection_revision", "curation_revision"))
    if revision < 3:
        raise ValueError("v2 is retired; generate fresh curated-v3 plans")
    retained = [row for row in (json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()) if row["split"] in ("train", "valid", "test")]
    if not retained or not all(row.get("oracle_derived") and row.get("cropPixelsXYWH") is not None and row.get("roi_sha256") for row in retained):
        raise ValueError("These experiment plans require the frozen oracle ROI v3 selection")
    code, dataset = PureWindowsPath(code_root), PureWindowsPath(data_root)
    conda = r"C:\Users\TRAINING_USER\miniforge3\Scripts\conda.exe"
    prefix = [conda, "run", "--no-capture-output", "-n", "mahjong-train", "python", "-m"]
    manifest_on_win = str(code / "bundle-inputs" / "split-manifest.jsonl")
    common = {"kind": "offline_experiment_plan", "launchable": False, "trainingStarted": False,
              "remoteAccessPerformed": False, "environment": "mahjong-train", "cwd": str(code),
              "dataset": str(dataset), "datasetMetadataSha256": digest(metadata),
              "trackedSplitSha256": digest(manifest), "archiveSha256": data["archive_sha256"],
              "freshRunRequired": True, "resumeFromV2Allowed": False, "productionValidated": False,
              "evaluationScope": "oracle_roi_auxiliary_only", "sealedTestUsedForSelection": False,
              "verifiedEnvironmentSnapshotSha256": digest(root / "infra/windows/verified/environment.json"),
              "verifiedPackageLockSha256": digest(root / "infra/windows/verified/requirements-lock.txt"),
              "notes": ["Windows is powered off; these are offline plans, not launched jobs.",
                        "Run the CPU metadata/image preflight after materialization, then a serial GPU smoke.",
                        "RF and RT must not run concurrently on the one 12 GB GPU.",
                        "30 epochs is a proposed upper bound; freeze all CLI/source/data fields before launch.",
                        "An oracle ROI validation score is not a user-cropped/full-photo or real-room accuracy claim."]}
    result = {}
    for family, module, batch, accumulation in (("rfdetr-small", "ml.train.rfdetr_train", 8, 2), ("rtdetr-r18", "ml.train.rtdetr_train", 2, 8)):
        smoke_name = f"{family}-640-curated-v3-smoke-seed42"
        full_name = f"{family}-640-curated-v3-seed42"
        source = root / Path(module.replace(".", "/")).with_suffix(".py")
        base = ["--dataset", str(dataset), "--data-manifest", manifest_on_win, "--resolution", "640", "--seed", "42"]
        if family == "rtdetr-r18":
            base += ["--source-directory", r"D:\mahjong\pretrained\rtdetr-v2-r18vd"]
        smoke = prefix + [module] + base + ["--output", str(PureWindowsPath(r"D:\mahjong\runs") / smoke_name),
                  "--epochs", "1", "--batch-size", "2" if family == "rfdetr-small" else "1", "--accumulation", "8" if family == "rfdetr-small" else "2", "--workers", "0"]
        if family == "rtdetr-r18":
            smoke += ["--train-limit", "32", "--valid-limit", "8"]
        full = prefix + [module] + base + ["--output", str(PureWindowsPath(r"D:\mahjong\runs") / full_name),
                  "--epochs", "30", "--batch-size", str(batch), "--accumulation", str(accumulation), "--workers", "2" if family == "rfdetr-small" else "0"]
        result[family] = dict(common, architecture=family, trainingSourcePath=source.relative_to(root).as_posix(), trainingSourceSha256=digest(source),
                             environmentVariables={"RF_HOME": r"D:\mahjong\models", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUTF8": "1"},
                             smokePreflightArgv=smoke + ["--preflight-only", "--verify-images"],
                             fullPreflightArgv=full + ["--preflight-only", "--verify-images"],
                             proposedSmokeArgv=smoke, proposedFullTrainingArgv=full,
                             nextBootStatus="pending_local_dataset_and_gpu_preflight")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows-code-root", default=r"D:\mahjong\code-curated-v3")
    parser.add_argument("--windows-data-root", default=r"D:\mahjong\data\curated-v3-roi")
    args = parser.parse_args()
    result = plans(args.root, args.metadata, args.split_manifest, args.windows_code_root, args.windows_data_root)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        (args.out / (name + ".plan.json")).write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps({"state": "offline_plans_written", "families": list(result), "trainingStarted": False, "remoteAccessPerformed": False}))


if __name__ == "__main__":
    main()
