"""Cache official YOLO and Torchvision weights without importing torch or training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

ASSETS = {
    **{name: {"url": f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}",
        "license": "AGPL-3.0", "source": "https://docs.ultralytics.com/models/yolo26/" if name.startswith("yolo26") else "https://docs.ultralytics.com/models/yolov8/"}
        for name in ("yolov8s.pt", "yolo26s.pt", "yolo26m.pt")},
    "efficientnet_v2_s-dd5fe13b.pth": {"url": "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth",
        "license": "Torchvision BSD-3-Clause; ImageNet pretrained weights terms apply",
        "source": "https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.efficientnet_v2_s.html",
        "expectedSha256Prefix": "dd5fe13b"}}


def cache(name, info, out):
    destination = out / name
    if not destination.is_file():
        temporary = destination.with_suffix(destination.suffix + ".part")
        curl = shutil.which("curl.exe") or shutil.which("curl")
        if curl:
            subprocess.run([curl, "--location", "--fail", "--retry", "2", "--max-time", "300",
                "--silent", "--show-error", "--output", str(temporary), info["url"]], check=True, timeout=1000)
        else:
            request = urllib.request.Request(info["url"], headers={"User-Agent": "mahjong-experiment-preparation/1"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
                while block := response.read(1024*1024):
                    stream.write(block)
        temporary.replace(destination)
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if info.get("expectedSha256Prefix") and not digest.startswith(info["expectedSha256Prefix"]):
        raise ValueError("Official Torchvision hash prefix mismatch")
    if destination.stat().st_size < 1_000_000:
        raise ValueError("Downloaded file is too small to be official pretrained weights")
    return dict(info, file=str(destination), bytes=destination.stat().st_size, sha256=digest,
                origin="official HTTPS release; SHA256 frozen from retrieved bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {name: pool.submit(cache, name, info, args.out) for name, info in ASSETS.items()}
        assets = {name: future.result() for name, future in futures.items()}
    manifest = {"retrievedAt": datetime.now(timezone.utc).isoformat(), "trainingStarted": False, "assets": assets}
    (args.out / "assets-yolo-cascade.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
