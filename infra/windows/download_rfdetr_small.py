"""Download the RF-DETR Small official weights into RF_HOME, with hash check."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess

URL = "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth"
SIZE = 386045550
MD5 = "fb37061c1af7bace359c91b723a8d5c1"
CHUNK = 8 * 1024**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path(r"D:\mahjong\models"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    final = args.directory / "rf-detr-small.pth"
    if final.exists() and final.stat().st_size == SIZE:
        data = final.read_bytes()
        if hashlib.md5(data).hexdigest() == MD5:
            print(f"Verified cached RF-DETR Small SHA256={hashlib.sha256(data).hexdigest()}", flush=True)
            return
    parts = args.directory / "rf-detr-small.parts"
    parts.mkdir(exist_ok=True)

    def fetch(index: int) -> Path:
        start, end = index * CHUNK, min((index + 1) * CHUNK, SIZE) - 1
        part = parts / f"{index:04d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        temp = part.with_suffix(".partial")
        subprocess.run(["curl.exe", "-4", "-fsSL", "--retry", "2", "--connect-timeout", "15", "--max-time", "180",
                        "--range", f"{start}-{end}", "--output", str(temp), URL], check=True)
        if temp.stat().st_size != end - start + 1:
            raise RuntimeError(f"Weight range {index} returned incorrect byte count")
        temp.replace(part)
        print(f"range {index+1}/{(SIZE+CHUNK-1)//CHUNK} complete", flush=True)
        return part

    with ThreadPoolExecutor(max_workers=8) as pool:
        downloaded = list(pool.map(fetch, range((SIZE + CHUNK - 1)//CHUNK)))
    temp = final.with_suffix(".assembling")
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with temp.open("wb") as out:
        for part in downloaded:
            block = part.read_bytes()
            out.write(block)
            md5.update(block)
            sha256.update(block)
    if temp.stat().st_size != SIZE or md5.hexdigest() != MD5:
        raise RuntimeError("Weight hash mismatch with the RF-DETR package registry")
    temp.replace(final)
    (final.with_suffix(".sha256")).write_text(sha256.hexdigest()+"\n", encoding="ascii")
    print(f"Verified RF-DETR Small SHA256={sha256.hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
