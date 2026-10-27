"""Offline dataset preflight and atomic preparation from the existing raw ZIP.

Default is a read-only dry run. --apply extracts/materializes only after archive,
selection and path checks. No network, SSH, GPU work or training is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from infra.windows.offline_bundle import digest, extract_new, inspect_zip, safe_name

SPLITS = ("train", "valid", "test")


def rows_from(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def check_selection(rows: list[dict], expected: dict) -> dict:
    seen, groups, counts = set(), {}, Counter()
    for row in rows:
        uid = safe_name(row["source_id"])
        if uid in seen:
            raise ValueError("Duplicate source ID in tracked split manifest")
        seen.add(uid)
        if row["split"] not in (*SPLITS, "excluded") or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")):
            raise ValueError("Invalid split or missing source SHA256")
        if row["split"] in SPLITS:
            group = row.get("group_id")
            if not group or groups.setdefault(group, row["split"]) != row["split"]:
                raise ValueError("Missing group or source group crossing retained splits")
            counts[row["split"]] += 1
    for split in SPLITS:
        if counts[split] != expected[split]["images"]:
            raise ValueError(f"Tracked {split} count differs from frozen metadata")
    return dict(counts)


def within(root: Path, relative: str) -> Path:
    name = safe_name(relative)
    path = root.joinpath(*name.split("/"))
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset path escapes its root")
    return path


def verify_original(original: Path, archive: Path, rows: list[dict]) -> None:
    # A stale _ready marker cannot conceal edited source annotations/images.
    with zipfile.ZipFile(archive) as zipped:
        for split in SPLITS:
            relative = f"{split}/_annotations.coco.json"
            if digest(within(original, relative)) != hashlib.sha256(zipped.read(relative)).hexdigest():
                raise ValueError(f"Extracted original COCO was modified: {split}")
    for row in rows:
        if row["split"] in SPLITS and digest(within(original, row["source_id"])) != row["sha256"]:
            raise ValueError(f"Source image SHA256 mismatch: {row['source_id']}")


def verify_prepared(prepared: Path, rows: list[dict], expected: dict) -> None:
    retained = {row["source_id"]: row for row in rows if row["split"] in SPLITS}
    seen = set()
    for split in SPLITS:
        annotation = prepared / split / "_annotations.coco.json"
        if digest(annotation) != expected[split]["annotation_sha256"]:
            raise ValueError(f"Prepared annotation differs from frozen snapshot: {split}")
        coco = json.loads(annotation.read_text(encoding="utf-8"))
        if len(coco["images"]) != expected[split]["images"] or len(coco["annotations"]) != expected[split]["annotations"]:
            raise ValueError(f"Prepared COCO counts differ: {split}")
        for image in coco["images"]:
            uid = image["source_id"]
            if uid in seen or uid not in retained or retained[uid]["split"] != split:
                raise ValueError("Prepared source image missing, duplicated or assigned to the wrong split")
            seen.add(uid)
            expected_image_sha = retained[uid].get("roi_sha256") if retained[uid].get("cropPixelsXYWH") is not None else retained[uid]["sha256"]
            if not expected_image_sha or digest(within(prepared / split, image["file_name"])) != expected_image_sha:
                raise ValueError(f"Prepared image SHA256 mismatch: {uid}")
    if seen != set(retained):
        raise ValueError("Prepared images do not cover the retained selection exactly")


def prepare(args) -> dict:
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    revision = max(int(metadata.get(key, 0)) for key in ("audit_revision", "selection_revision", "curation_revision"))
    if revision < 3 and not args.allow_research_v2:
        raise ValueError("Use the reviewed v3 selection; legacy v2 requires explicit --allow-research-v2 for research only")
    expected_archive = metadata["archive_sha256"]
    expected_split = metadata["tracked_split_sha256"]
    if digest(args.archive) != expected_archive or digest(args.split_manifest) != expected_split:
        raise ValueError("Archive or tracked selection SHA256 mismatch")
    entries = inspect_zip(args.archive)
    member_names = {safe_name(entry.orig_filename) for entry in entries if not entry.is_dir()}
    rows = rows_from(args.split_manifest)
    counts = check_selection(rows, metadata["prepared_splits"])
    retained = [row for row in rows if row["split"] in SPLITS]
    oracle_roi = any(row.get("cropPixelsXYWH") is not None or row.get("oracle_derived") for row in retained)
    if oracle_roi and not all(row.get("cropPixelsXYWH") is not None and row.get("oracle_derived") is True and re.fullmatch(r"[0-9a-f]{64}", row.get("roi_sha256", "")) for row in retained):
        raise ValueError("Oracle ROI selection must provide crop, provenance and derivative SHA256 for every retained image")
    if oracle_roi:
        for row in retained:
            crop = row["cropPixelsXYWH"]
            if not isinstance(crop, list) or len(crop) != 4 or not all(isinstance(value, int) and not isinstance(value, bool) for value in crop) or min(crop[:2]) < 0 or min(crop[2:]) <= 0:
                raise ValueError("Oracle ROI must be a bounded positive integer XYWH crop")
    if not {row["source_id"] for row in rows}.issubset(member_names):
        raise ValueError("Tracked source IDs are missing from the immutable archive")
    for value in (args.original_name, args.prepared_name):
        if "/" in safe_name(value):
            raise ValueError("Original/prepared names must be single directory names")
    original = args.root / args.original_name
    prepared = args.root / args.prepared_name
    metadata_sha, split_sha = digest(args.metadata), digest(args.split_manifest)
    report = {"state": "dry_run_verified", "dryRun": not args.apply, "archiveSha256": expected_archive,
              "trackedSplitSha256": split_sha, "metadataSha256": metadata_sha, "selectionRevision": revision,
              "selectedImages": counts, "original": str(original), "prepared": str(prepared),
              "trainingStarted": False, "remoteAccessPerformed": False, "productionValidated": False,
              "preparedHashes": "pending_materialization",
              "evaluationScope": "oracle_roi_auxiliary_only" if oracle_roi else "research_annotation_scope_unverified",
              "freshExperimentRequired": True, "resumeFromV2Allowed": False}
    if oracle_roi:
        # Read only the selected source members; the immutable ZIP can be reused
        # without extracting a second multi-gigabyte copy of the whole dataset.
        with zipfile.ZipFile(args.archive) as zipped:
            for row in retained:
                with zipped.open(row["source_id"]) as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                        raise ValueError(f"Source member SHA256 mismatch: {row['source_id']}")
        report["selectedSourceImageHashes"] = "verified_from_archive"
    elif original.exists():
        marker = original / "_ready.json"
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8-sig")).get("archiveSha256") != expected_archive:
            raise ValueError("Existing original directory lacks a matching verified archive marker")
        verify_original(original, args.archive, rows)
        report["selectedSourceImageHashes"] = "verified"
    else:
        report["selectedSourceImageHashes"] = "pending_extraction"
    if prepared.exists():
        marker = prepared / "_ready.json"
        value = json.loads(marker.read_text(encoding="utf-8-sig")) if marker.is_file() else {}
        if value.get("trackedSplitSha256") != split_sha or value.get("metadataSha256") != metadata_sha:
            raise ValueError("Existing prepared directory does not match this exact selection; choose a fresh directory")
        verify_prepared(prepared, rows, metadata["prepared_splits"])
        report["preparedHashes"] = "verified"
    if not args.apply:
        return report
    if not oracle_roi and not original.exists():
        extract_new(args.archive, original, expected_sha256=expected_archive)
        verify_original(original, args.archive, rows)
        (original / "_ready.json").write_text(json.dumps({"archiveSha256": expected_archive}), encoding="utf-8")
    if not prepared.exists():
        args.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{prepared.name}.materialize-", dir=args.root))
        manifest = staging.parent / (staging.name + ".jsonl")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        try:
            if oracle_roi:
                manifest.write_bytes(args.split_manifest.read_bytes().replace(b"\r\n", b"\n"))
                subprocess.run([sys.executable, "-m", "ml.data.materialize_v3", "--archive", str(args.archive),
                                "--split-manifest", str(args.split_manifest), "--out", str(staging)], cwd=REPO, env=env, check=True)
            else:
                subprocess.run([sys.executable, "-m", "ml.data.rebuild_manifest", "--root", str(original),
                                "--split-manifest", str(args.split_manifest), "--out", str(manifest)], cwd=REPO, env=env, check=True)
                manifest.write_bytes(manifest.read_bytes().replace(b"\r\n", b"\n"))
                if metadata.get("manifest_sha256") and digest(manifest) != metadata["manifest_sha256"]:
                    raise ValueError("Rebuilt materialization manifest SHA256 differs from frozen metadata")
                subprocess.run([sys.executable, "-m", "ml.data.materialize", "--root", str(original),
                                "--manifest", str(manifest), "--out", str(staging)], cwd=REPO, env=env, check=True)
            verify_prepared(staging, rows, metadata["prepared_splits"])
            ready = {"archiveSha256": expected_archive, "trackedSplitSha256": split_sha,
                     "metadataSha256": metadata_sha, "manifestSha256": digest(manifest),
                     "selectionRevision": revision, "productionValidated": False,
                     "manifestFile": "_selection-manifest.jsonl" if oracle_roi else "_materialization-manifest.jsonl",
                     "evaluationScope": report["evaluationScope"], "freshExperimentRequired": True, "resumeFromV2Allowed": False}
            (staging / "_ready.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
            staging.replace(prepared)
            manifest.replace(prepared / ready["manifestFile"])
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            if manifest.exists():
                manifest.unlink()
            raise
    report.update(state="prepared_verified", selectedSourceImageHashes="verified", preparedHashes="verified")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--data-root", type=Path, default=Path("D:/mahjong/data"))
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--original-name", default="roboflow-v2-original")
    parser.add_argument("--prepared-name", default="curated-v3-roi")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-research-v2", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2))


if __name__ == "__main__":
    main()
