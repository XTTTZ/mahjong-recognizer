"""Offline CPU-only compatibility check. No optimizer, training or GPU computation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--training-options', type=Path, required=True,
                        help='JSON captured from the real training_options function, used only by get_cfg')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new compatibility evidence filename')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Empty values can remove an environment variable on Windows; -1 masks all GPUs.
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false',
                      YOLO_CONFIG_DIR=str(args.output.parent / 'yolo-cpu-settings'),
                      HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    report = {'observedAt': datetime.now(timezone.utc).isoformat(), 'trainingStarted': False,
              'backwardCalled': False, 'gpuComputation': False, 'cudaVisibleDevices': '-1',
              'scriptSha256': sha256(__file__), 'models': []}
    try:
        import torch
        import torchvision
        import cv2
        import numpy as np
        from PIL import Image
        from ultralytics import YOLO, settings
        from ultralytics.cfg import get_cfg
        if importlib.metadata.version('ultralytics') != '8.4.189':
            raise RuntimeError('Expected the pinned Ultralytics 8.4.189')
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            raise RuntimeError('CPU readiness must hide CUDA devices before torch import')
        settings.update({'sync': False, 'wandb': False, 'mlflow': False, 'clearml': False,
                         'comet': False, 'dvc': False, 'raytune': False, 'tensorboard': False})
        report['runtime'] = {'torch': torch.__version__, 'torchvision': torchvision.__version__,
                             'ultralytics': importlib.metadata.version('ultralytics'),
                             'cv2': cv2.__version__, 'numpy': np.__version__}
        assert cv2.cvtColor(np.zeros((2, 2, 3), dtype=np.uint8), cv2.COLOR_RGB2GRAY).shape == (2, 2)
        payload = json.loads(args.training_options.read_text(encoding='utf-8-sig'))
        schema = get_cfg(overrides=payload['trainingOptions'])
        assert schema.amp == 'bf16' and schema.cls_remap is False and schema.augmentations == []
        report['trainingSchema'] = {'passed': True, 'optionsSha256': sha256(args.training_options),
                                    'sourceSha256': payload['trainingSourceSha256'],
                                    'validatedKeys': sorted(payload['trainingOptions'])}
        image = Image.new('RGB', (320, 320), (114, 114, 114))
        for filename in ('yolov8s.pt', 'yolo26s.pt', 'yolo26m.pt'):
            weights = args.weights_dir / filename
            started = time.perf_counter()
            model = YOLO(str(weights), task='detect').to('cpu')
            assert len(model.names) == 80
            assert all(parameter.device.type == 'cpu' for parameter in model.model.parameters())
            with torch.inference_mode():
                result = model.predict(image, imgsz=320, device='cpu', conf=.05, iou=.6,
                    max_det=300, quantize=32, verbose=False, augment=False, rect=False, save=False)[0]
            assert result.boxes.xyxy.device.type == 'cpu'
            assert torch.isfinite(result.boxes.xyxy).all() and torch.isfinite(result.boxes.conf).all()
            report['models'].append({'file': filename, 'sha256': sha256(weights),
                'sourceClasses': len(model.names), 'predictionCount': len(result.boxes),
                'inputResolution': 320, 'device': 'cpu', 'finite': True,
                'elapsedSeconds': round(time.perf_counter()-started, 3)})
            del result, model
        from torch import nn
        from torchvision.models import efficientnet_v2_s
        weights = args.weights_dir / 'efficientnet_v2_s-dd5fe13b.pth'
        started = time.perf_counter()
        classifier = efficientnet_v2_s(weights=None).eval()
        classifier.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        inputs = torch.zeros((1, 3, 320, 320), device='cpu')
        with torch.inference_mode():
            original = classifier(inputs)
            assert tuple(original.shape) == (1, 1000) and torch.isfinite(original).all()
            classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, 42)
            classifier.eval()
            replaced = classifier(inputs)
            assert tuple(replaced.shape) == (1, 42) and torch.isfinite(replaced).all()
        assert all(parameter.device.type == 'cpu' for parameter in classifier.parameters())
        report['classifier'] = {'file': weights.name, 'sha256': sha256(weights),
            'sourceOutputShape': list(original.shape), 'replacementOutputShape': list(replaced.shape),
            'inputResolution': 320, 'device': 'cpu', 'finite': True,
            'elapsedSeconds': round(time.perf_counter()-started, 3)}
        report['passed'] = True
    except Exception as error:
        report.update(passed=False, errorType=type(error).__name__, error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
