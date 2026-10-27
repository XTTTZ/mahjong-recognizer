"""Fetch the pinned official Windows CUDA wheel with resumable range parts.

IPv4 avoids a slow IPv6 route observed on this host. Every range must have the
expected byte count; the final file is checked against the official index hash.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import subprocess

URL = "https://download-r2.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp311-cp311-win_amd64.whl"
SIZE = 3461420395
SHA256 = "34c55443aafd31046a7963b63d30bc3b628ee4a704f826796c865fdfd05bb596"
CHUNK = 64 * 1024**2


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    final = args.directory / "torch-2.8.0+cu128-cp311-cp311-win_amd64.whl"
    if final.exists() and final.stat().st_size == SIZE and digest(final) == SHA256:
        print("Official torch wheel already verified", flush=True)
        return
    parts = args.directory / "torch-2.8.0-cu128.parts"
    parts.mkdir(exist_ok=True)

    def fetch(index: int) -> Path:
        start, end = index * CHUNK, min((index + 1) * CHUNK, SIZE) - 1
        part = parts / f"{index:04d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        partial = part.with_suffix(".partial")
        subprocess.run(["curl.exe", "-4", "--fail", "--silent", "--show-error", "--location", "--retry", "2",
                        "--connect-timeout", "20", "--max-time", "600", "--range", f"{start}-{end}", "--output", str(partial), URL], check=True)
        if partial.stat().st_size != end - start + 1:
            raise RuntimeError(f"Range {index} returned incorrect byte length")
        partial.replace(part)
        print(f"range {index + 1}/{(SIZE + CHUNK - 1)//CHUNK} complete", flush=True)
        return part

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        downloaded = list(pool.map(fetch, range((SIZE + CHUNK - 1) // CHUNK)))
    assembling = final.with_suffix(".assembling")
    with assembling.open("wb") as out:
        for part in downloaded:
            with part.open("rb") as stream:
                shutil.copyfileobj(stream, out, 8 * 1024**2)
    if assembling.stat().st_size != SIZE or digest(assembling) != SHA256:
        raise RuntimeError("Official torch SHA256 mismatch; do not install")
    assembling.replace(final)
    print(f"SHA256 verified: {final}", flush=True)


if __name__ == "__main__":
    main()
