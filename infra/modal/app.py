"""Deploy a frozen, evaluated model: MAHJONG_MODEL_DIR=/... python -m modal deploy infra/modal/app.py.

The local model directory MUST contain model.json and its SHA256-matching
checkpoint. No training data or credentials are included in the image.
"""
import os
import json
import sys
from pathlib import Path
import modal

app = modal.App(os.environ.get("MAHJONG_MODAL_APP", "mahjong-inference-staging"))
image = None
RUNTIME = os.environ.get('MAHJONG_RUNTIME', 'torch')
if RUNTIME not in {'torch', 'tensorrt'}:
    raise ValueError('MAHJONG_RUNTIME must be torch or tensorrt')
SNAPSHOT = os.environ.get('MAHJONG_SNAPSHOT', '1') == '1'
GPU_TYPES = [name.strip() for name in os.environ.get("MAHJONG_GPU", "L4,A10").split(",") if name.strip()]
if not GPU_TYPES:
    raise ValueError("MAHJONG_GPU must include at least one GPU type")
if RUNTIME == 'tensorrt' and (len(GPU_TYPES) != 1 or GPU_TYPES[0] not in {'L4', 'A10'}):
    raise ValueError('TensorRT engines must be built and served on the same single L4 or A10 GPU type')
# Modal imports this module as /root/app.py inside containers. Only the local
# deploy process may inspect workspace files or construct the image.
if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    MODEL_DIR = os.environ.get("MAHJONG_MODEL_DIR")
    if not MODEL_DIR or not (Path(MODEL_DIR) / "model.json").is_file():
        raise RuntimeError("Set MAHJONG_MODEL_DIR to a frozen model directory containing model.json")

    sys.path.insert(0, str(ROOT))
    from ml.inference.model import inspect_model
    from infra.modal.runtime import packages_for, runtime_environment
    from ml.inference.manifest import file_sha256
    from infra.modal.package import prepare_upload, prepare_source
    manifest = inspect_model(MODEL_DIR)
    prebuilt_image = os.environ.get('MAHJONG_PREBUILT_IMAGE_ID')
    if prebuilt_image:
        source_dir = prepare_source(ROOT/'ml/inference', ROOT/'artifacts/models/modal-inference-source')
        image = (modal.Image.from_id(prebuilt_image)
            .add_local_dir(source_dir, remote_path='/app/ml/inference', copy=True))
    else:
        model_packages = packages_for(manifest["architecture"])
        if RUNTIME == 'tensorrt':
            if manifest['architecture'] != 'yolo-efficientnet-v2-s-cascade':
                raise ValueError('TensorRT currently supports the reviewed cascade only')
            model_packages += ['tensorrt-cu12==10.13.3.9', 'onnx==1.19.0']
        upload_dir = prepare_upload(MODEL_DIR, ROOT / "artifacts/models/modal-upload")
        source_dir = prepare_source(ROOT/'ml', ROOT/'artifacts/models/modal-source')

        image = (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install("libgl1", "libglib2.0-0")
            .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
            .pip_install(*model_packages)
            .add_local_dir(upload_dir, remote_path="/model-upload", copy=True)
            .run_commands("python /model-upload/assemble.py && rm -rf /model-upload")
            .env({"PYTHONPATH": "/app", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                  "YOLO_CONFIG_DIR": "/tmp/mahjong-yolo", "YOLO_AUTOINSTALL": "false",
                  'MAHJONG_RUNTIME': RUNTIME})
            .add_local_dir(source_dir, remote_path="/app/ml", copy=True)
        )
        if RUNTIME == 'tensorrt':
            from ml.inference.tensorrt_runtime import build_engines
            image = image.run_function(build_engines, gpu=GPU_TYPES[0], cpu=2, memory=8192, timeout=1800)
    image = image.env(runtime_environment(RUNTIME, GPU_TYPES, SNAPSHOT,
        manifest['version'], file_sha256(Path(MODEL_DIR)/'model.json')))


# Both GPUs support the classifier's BF16 inference. A second eligible type
# avoids waiting exclusively for L4 capacity during a cold request.
@app.cls(image=image, gpu=GPU_TYPES, min_containers=0, max_containers=1,
         cpu=2, memory=4096, scaledown_window=30, timeout=90,
         enable_memory_snapshot=SNAPSHOT,
         experimental_options={'enable_gpu_snapshot': True} if SNAPSHOT else {})
class Recognizer:
    @modal.enter(snap=SNAPSHOT)
    def load(self):
        # A reused build image may also contain build-time Python modules in
        # /root; the serving adapter always comes from the frozen /app overlay.
        sys.path.insert(0, '/app')
        import torch
        # Match CPU parallelism to the allocated two cores, before warmup.
        torch.set_num_threads(2)
        from ml.inference.model import load_predictor
        self.predictor = load_predictor("/model")
        expected = json.loads(os.environ['MAHJONG_EXPECTED_MODEL'])
        from ml.inference.manifest import file_sha256
        if (self.predictor.manifest['version'] != expected['version']
                or file_sha256('/model/model.json') != expected['manifestSha256']):
            raise ValueError('Prebuilt image does not contain the selected frozen model')
        if hasattr(self.predictor, 'enable_fast_preprocess'):
            self.predictor.enable_fast_preprocess()
        if RUNTIME == 'tensorrt':
            from ml.inference.tensorrt_runtime import activate_tensorrt
            activate_tensorrt(self.predictor)
        if hasattr(self.predictor, 'warmup'):
            self.predictor.warmup()

    @modal.asgi_app(requires_proxy_auth=True)
    def api(self):
        from ml.inference.service import create_app
        return create_app(self.predictor, self.predictor.manifest.get("acceptThreshold", .7))
