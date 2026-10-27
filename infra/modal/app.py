"""Deploy a frozen, evaluated model: MAHJONG_MODEL_DIR=/... python -m modal deploy infra/modal/app.py.

The local model directory MUST contain model.json and its SHA256-matching
checkpoint. No training data or credentials are included in the image.
"""
import os
import sys
from pathlib import Path
import modal

app = modal.App(os.environ.get("MAHJONG_MODAL_APP", "mahjong-inference-staging"))
image = None
# Modal imports this module as /root/app.py inside containers. Only the local
# deploy process may inspect workspace files or construct the image.
if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    MODEL_DIR = os.environ.get("MAHJONG_MODEL_DIR")
    if not MODEL_DIR or not (Path(MODEL_DIR) / "model.json").is_file():
        raise RuntimeError("Set MAHJONG_MODEL_DIR to a frozen model directory containing model.json")

    sys.path.insert(0, str(ROOT))
    from ml.inference.model import inspect_model
    from infra.modal.runtime import packages_for
    from infra.modal.package import prepare_upload
    manifest = inspect_model(MODEL_DIR)
    model_packages = packages_for(manifest["architecture"])
    upload_dir = prepare_upload(MODEL_DIR, ROOT / "artifacts/models/modal-upload")

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("libgl1", "libglib2.0-0")
        .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
        .pip_install(*model_packages)
        .add_local_dir(upload_dir, remote_path="/model-upload", copy=True)
        .run_commands("python /model-upload/assemble.py && rm -rf /model-upload")
        .env({"PYTHONPATH": "/app", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
              "YOLO_CONFIG_DIR": "/tmp/mahjong-yolo", "YOLO_AUTOINSTALL": "false"})
        .add_local_dir(ROOT / "ml", remote_path="/app/ml", ignore=["**/__pycache__/**"])
    )


@app.cls(image=image, gpu=os.environ.get("MAHJONG_GPU", "L4"), min_containers=0, max_containers=1,
         cpu=1, memory=4096, scaledown_window=15, timeout=60)
class Recognizer:
    @modal.enter()
    def load(self):
        from ml.inference.model import load_predictor
        self.predictor = load_predictor("/model")

    @modal.asgi_app(requires_proxy_auth=True)
    def api(self):
        from ml.inference.service import create_app
        return create_app(self.predictor, self.predictor.manifest.get("acceptThreshold", .7))
