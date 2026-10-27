"""Build/verify/unpack a small offline training code snapshot; never uses SSH.

Large datasets, model weights, credentials and frontend dependencies are excluded.
The expected archive SHA256 is required before unpacking a transported bundle.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import subprocess
import tempfile
import unicodedata
import zipfile

MANIFEST = "BUNDLE-MANIFEST.json"
RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"{p}{n}" for p in ("COM", "LPT") for n in range(1, 10)}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def safe_name(name: str) -> str:
    """Reject POSIX and Windows escape/alias paths, including NTFS streams."""
    if not name or "\\" in name or "\x00" in name or name.startswith("/") or PureWindowsPath(name).drive:
        raise ValueError(f"Unsafe archive path: {name!r}")
    name = name[:-1] if name.endswith("/") else name
    parts = name.split("/")
    if not name or any(p in ("", ".", "..") or ":" in p or p.endswith((".", " ")) or p.split(".", 1)[0].upper() in RESERVED for p in parts):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return PurePosixPath(name).as_posix()


def inspect_zip(archive: Path, *, max_files: int = 100000, max_total_bytes: int = 20 * 1024**3) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        if len(entries) > max_files or sum(e.file_size for e in entries) > max_total_bytes:
            raise ValueError("ZIP exceeds the configured extraction limits")
        names: dict[str, bool] = {}
        for entry in entries:
            path = safe_name(entry.orig_filename)
            key = unicodedata.normalize("NFC", path).casefold()
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"ZIP special file/symlink is forbidden: {path}")
            if entry.file_size > 512 * 1024**2 or entry.flag_bits & 1:
                raise ValueError(f"Unsupported large/encrypted ZIP member: {path}")
            if key in names:
                raise ValueError(f"ZIP has duplicate/case-aliased paths: {path}")
            names[key] = entry.is_dir()
        for key in names:
            for parent in PurePosixPath(key).parents:
                if str(parent) != "." and str(parent) in names and not names[str(parent)]:
                    raise ValueError("ZIP places a member beneath a regular file")
        return entries


def extract_new(archive: Path, destination: Path, *, expected_sha256: str, max_total_bytes: int = 20 * 1024**3) -> None:
    if digest(archive) != expected_sha256:
        raise ValueError("Archive SHA256 mismatch")
    entries = inspect_zip(archive, max_total_bytes=max_total_bytes)
    if destination.exists():
        raise FileExistsError("Extraction destination exists; choose a fresh directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.extract-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as source:
            for entry in entries:
                target = staging.joinpath(*safe_name(entry.orig_filename).split("/"))
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(entry) as incoming, target.open("xb") as output:
                        shutil.copyfileobj(incoming, output, 1024**2)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise


def payload(root: Path) -> dict[str, bytes]:
    files: set[Path] = set()
    extensions = {".py", ".ps1", ".json", ".jsonl", ".txt", ".yml", ".yaml", ".md"}
    for directory in (root / "ml", root / "infra" / "windows", root / "docs" / "data"):
        if directory.exists():
            files.update(p for p in directory.rglob("*") if p.is_file() and p.suffix in extensions and "__pycache__" not in p.parts)
    for name in ("scripts/prepare_windows_data.py", "NOTICE.md", "PLAN.md"):
        path = root / name
        if path.is_file():
            files.add(path)
    result = {}
    for path in sorted(files):
        if "specs" in path.relative_to(root).parts:
            # Historical launch specs and oracle-v3 plans remain in Git. Fresh
            # ready suites are generated separately after code/data/env freeze;
            # importing stale plans here can silently bind an obsolete recipe.
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Code snapshot contains a symlink/outside file")
        name = safe_name(path.relative_to(root).as_posix())
        data = path.read_bytes()
        if len(data) > 20 * 1024**2:
            raise ValueError(f"Unexpected large code/metadata file: {name}")
        result[name] = data
    rt_source = root / "artifacts/models/pretrained/rtdetr-v2-r18vd/source.json"
    if rt_source.is_file():
        result["bundle-inputs/rtdetr-source.json"] = rt_source.read_bytes()
    if sum(map(len, result.values())) > 64 * 1024**2:
        raise ValueError("Code snapshot exceeds 64 MiB; do not include datasets/weights")
    return result


def build(root: Path, out: Path, metadata: Path | None, split_manifest: Path | None) -> dict:
    if bool(metadata) != bool(split_manifest):
        raise ValueError("Provide both dataset metadata and tracked split manifest, or neither")
    contents = payload(root)
    dataset = None
    if metadata:
        value = json.loads(metadata.read_text(encoding="utf-8"))
        expected = value.get("tracked_split_sha256")
        if not expected or digest(split_manifest) != expected:
            raise ValueError("Dataset metadata and tracked split manifest do not match")
        contents["bundle-inputs/dataset-metadata.json"] = metadata.read_bytes()
        contents["bundle-inputs/split-manifest.jsonl"] = split_manifest.read_bytes()
        dataset = {"metadataSha256": digest(metadata), "trackedSplitSha256": expected, "archiveSha256": value["archive_sha256"],
                   "archiveIncluded": False, "note": "Reuse the verified immutable v2 source archive; apply the new v3 selection only."}
    for name, data in contents.items():
        if name.endswith(".plan.json"):
            plan = json.loads(data)
            source = plan.get("trainingSourcePath")
            if source not in contents or hashlib.sha256(contents[source]).hexdigest() != plan.get("trainingSourceSha256"):
                raise ValueError(f"Source changed after experiment plan generation: {name}")
            for path, field in (("infra/windows/verified/environment.json", "verifiedEnvironmentSnapshotSha256"),
                                ("infra/windows/verified/requirements-lock.txt", "verifiedPackageLockSha256")):
                if hashlib.sha256(contents[path]).hexdigest() != plan.get(field):
                    raise ValueError(f"Environment snapshot changed after plan generation: {name}")
            if dataset and (dataset["metadataSha256"] != plan.get("datasetMetadataSha256") or dataset["trackedSplitSha256"] != plan.get("trackedSplitSha256")):
                raise ValueError(f"Dataset changed after experiment plan generation: {name}")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, True
    manifest = {"schema_version": 1, "kind": "offline-training-code-snapshot", "generated_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": commit, "working_tree_dirty": dirty, "training_started": False, "remote_access_performed": False,
                "dataset": dataset, "remote_preflight_required": True,
                "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data in sorted(contents.items())}}
    contents[MANIFEST] = json.dumps(manifest, indent=2).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            zipped.writestr(info, data)
    result = {"bundle": str(out), "sha256": digest(out), "bytes": out.stat().st_size, "payload_files": len(manifest["files"]),
              "dataset_inputs_included": dataset is not None, "training_started": False}
    out.with_suffix(out.suffix + ".sha256").write_text(result["sha256"] + "\n", encoding="ascii")
    verifier = out.with_suffix(".verify.py")
    verifier.write_bytes(Path(__file__).read_bytes())
    result.update(verifier=str(verifier), verifier_sha256=digest(verifier))
    out.with_suffix(".receipt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def verify(bundle: Path, expected_sha256: str) -> dict:
    if digest(bundle) != expected_sha256:
        raise ValueError("Bundle SHA256 does not match trusted expected value")
    entries = inspect_zip(bundle, max_total_bytes=64 * 1024**2)
    with zipfile.ZipFile(bundle) as source:
        manifest = json.loads(source.read(MANIFEST))
        actual = {entry.filename for entry in entries if not entry.is_dir()} - {MANIFEST}
        if actual != set(manifest["files"]):
            raise ValueError("Bundle payload differs from its file manifest")
        for name, expected in manifest["files"].items():
            data = source.read(name)
            if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise ValueError(f"Bundle file hash mismatch: {name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path.cwd())
    create.add_argument("--out", type=Path, required=True)
    create.add_argument("--dataset-metadata", type=Path)
    create.add_argument("--split-manifest", type=Path)
    for name in ("verify", "unpack"):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--expected-sha256", required=True)
        if name == "unpack":
            command.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.root, args.out, args.dataset_metadata, args.split_manifest)
    else:
        manifest = verify(args.bundle, args.expected_sha256)
        if args.command == "unpack":
            extract_new(args.bundle, args.destination, expected_sha256=args.expected_sha256, max_total_bytes=64 * 1024**2)
        result = {"state": "verified" if args.command == "verify" else "unpacked", "files": len(manifest["files"]), "dataset": manifest["dataset"],
                  "training_started": False, "remote_access_performed": False, "remote_preflight_required": True}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
