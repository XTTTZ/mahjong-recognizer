"""Research-only small classifier cascade; production model factory stays unchanged."""
import json
from pathlib import Path
from ml.inference.cascade_model import RawCascade
from ml.train.student_common import ARCHITECTURE, build_student


class RawStudentCascade(RawCascade):
    def __init__(self, detector, checkpoint, classes, config, device='cpu', scripted=None):
        import torch
        self.detector, self.classes, self.device = detector, classes, device
        self.resolution, self.context = config['resolution'], config['context']
        self.fast_preprocess = False
        self.runtime_backend = ('torch-bf16' if device == 'cuda' else 'torch-fp32') if scripted is None else 'cpu-int8'
        if scripted is not None:
            if device != 'cpu': raise ValueError('INT8 research artifacts support CPU only')
            self.classifier = torch.jit.load(str(scripted), map_location='cpu').eval()
        else:
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if saved['classes'] != classes or saved['config'] != config or saved.get('architecture') != ARCHITECTURE:
                raise ValueError('Student tensor artifact differs from signed geometry/classes/config')
            self.classifier = build_student()
            self.classifier.load_state_dict(saved['model'], strict=True)
            self.classifier.to(device).eval()
