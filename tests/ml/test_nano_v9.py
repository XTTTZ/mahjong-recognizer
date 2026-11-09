"""Nano preparation and shared completion receipts; no real model or GPU execution."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

from tests.ml.test_v6_preparation import fixture, review
from ml.data.prepare_yolo import prepare as prepare_yolo
from ml.data.prepare_background_v6 import prepare as prepare_background
from ml.eval.yolo_evaluate import run_metadata
from ml.train.provenance import sha256
from ml.train.yolo_train import enforce_fixed_batch, verify_receipt
from ml.train.yolo_train_v6 import train_bounded_detector
from ml.train.yolo_train_v9 import options, validate_recipe, initialization_contract, reviewed_derivative
from ml.train import yolo_train_v9


def args(root):
    return argparse.Namespace(dataset=root/'data', prepared=root/'prepared', weights=root/'yolov8n.pt',
        background_review=root/'review.json', output=root/'output', resolution=800, epochs=45,
        batch=8, workers=2, seed=2026, lr=.0003, augmentation='photometric', geometry='mild',
        max_train_minutes=85, preflight_only=True)


class NanoPreparationTests(unittest.TestCase):
    def test_reviewed_options_keep_nano_fresh_schedule_and_fixed_batch(self):
        a = args(Path('/fixture')); validate_recipe(a); cfg = options(a)
        self.assertEqual((cfg['imgsz'], cfg['batch'], cfg['warmup_epochs']), (800, 8, 3))
        self.assertEqual((cfg['amp'], cfg['patience'], cfg['degrees']), ('bf16', 0, 8.))
        self.assertEqual((cfg['flipud'], cfg['fliplr'], cfg['mosaic']), (0., 0., 0.))
        for key, value in [('batch', 4), ('resolution', 832), ('epochs', 0), ('lr', float('nan')),
                           ('max_train_minutes', float('inf')), ('max_train_minutes', -1)]:
            bad = args(Path('/fixture')); setattr(bad, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError): validate_recipe(bad)
        fake = SimpleNamespace(batch_size=8, args=SimpleNamespace(batch=8))
        enforce_fixed_batch(fake, 8); self.assertEqual(fake._oom_retries, 3)
        fake.batch_size = 4
        with self.assertRaises(RuntimeError): enforce_fixed_batch(fake, 8)

    def test_initialization_hash_and_filename_are_both_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'yolov8n.pt'; path.write_bytes(b'fixture-not-a-model')
            with patch.object(yolo_train_v9, 'WEIGHTS_SHA256', sha256(path)):
                self.assertFalse(initialization_contract(path)['resume'])
                wrong = path.with_name('yolov8s.pt'); wrong.write_bytes(path.read_bytes())
                with self.assertRaises(ValueError): initialization_contract(wrong)
                path.write_bytes(b'tampered')
                with self.assertRaises(ValueError): initialization_contract(path)

    def test_preflight_uses_reviewed_train_only_derivative_without_loading_models(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); a = args(root); contract = fixture(a.dataset)
            prepare_yolo(a.dataset, root/'base', True)
            a.background_review.write_text(json.dumps(review(a.dataset, contract)))
            prepare_background(a.dataset, root/'base', a.background_review, a.prepared)
            a.weights.write_bytes(b'no-model-can-be-loaded')
            output = io.StringIO()
            with patch.object(argparse.ArgumentParser, 'parse_args', return_value=a), \
                 patch.object(yolo_train_v9, 'WEIGHTS_SHA256', sha256(a.weights)), \
                 patch.dict(sys.modules, {'torch': None, 'ultralytics': None}), \
                 redirect_stdout(output):
                yolo_train_v9.main()
            receipt = json.loads(output.getvalue())
            self.assertTrue(receipt['derivativeReady']); self.assertFalse(receipt['trainingStarted'])
            self.assertFalse(receipt['testOpened']); self.assertEqual(receipt['backgrounds'], 1)
            self.assertEqual(receipt['initialization']['kind'], 'official_coco_pretrained')
            self.assertFalse(a.output.exists())
            # A plain positive-only derivative must not masquerade as reviewed.
            with self.assertRaises(ValueError): reviewed_derivative(root/'base', a.background_review, contract)
            self.assertIsNone(reviewed_derivative(root/'missing', a.background_review, contract))
            tampered = json.loads(a.background_review.read_text()); tampered['contactSheetsVisuallyReviewed'] = False
            a.background_review.write_text(json.dumps(tampered))
            with self.assertRaises(ValueError): reviewed_derivative(a.prepared, a.background_review, contract)

    def test_mocked_complete_epoch_has_evaluator_compatible_nano_receipt(self):
        # Exercise the real shared callbacks with inert byte checkpoints; there
        # is deliberately no torch tensor/model, optimizer, or training here.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); a = args(root); a.epochs = 1
            contract = fixture(a.dataset)
            signature = {'config': {'fixture': True}, 'classes': contract['classes'], 'data': contract['splits']}
            run = {'architecture': 'yolov8n-generic', 'generic': True, 'classes': contract['classes'],
                   'resolution': 800, 'resumeSignature': signature,
                   'annotationSha256': contract['splits']['train']['annotationSha256'],
                   'validationAnnotationSha256': contract['splits']['valid']['annotationSha256']}
            torch = ModuleType('torch')
            torch.cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True)
            ultralytics = ModuleType('ultralytics'); ultralytics.settings = {}
            data = ModuleType('ultralytics.data'); utils = ModuleType('ultralytics.data.utils')
            utils.check_font = lambda: None; data.utils = utils; ultralytics.data = data
            class MockYOLO:
                def __init__(self, checkpoint, task):
                    self.callbacks = {}
                def add_callback(self, name, callback): self.callbacks[name] = callback
                def train(self, **kwargs):
                    folder = a.output/'train/weights'; folder.mkdir(parents=True)
                    last, best = folder/'last.pt', folder/'best.pt'
                    last.write_bytes(b'last'); best.write_bytes(b'best')
                    csv = a.output/'train/results.csv'
                    csv.write_text('epoch,train/box_loss,metrics/mAP50(B),metrics/mAP50-95(B)\n1,1.5,.8,.6\n')
                    self.trainer = SimpleNamespace(last=last, best=best, epoch=0, epoch_time=1., csv=csv,
                        batch_size=8, args=SimpleNamespace(batch=8), accumulate=4, stop=False)
                    for name in ('on_train_epoch_start', 'on_model_save', 'on_fit_epoch_end'):
                        self.callbacks[name](self.trainer)
            ultralytics.YOLO = MockYOLO
            with patch.dict(sys.modules, {'torch': torch, 'ultralytics': ultralytics,
                       'ultralytics.data': data, 'ultralytics.data.utils': utils}), \
                 patch('ml.train.yolo_train_v6.importlib.metadata.version', return_value='8.4.189'), \
                 patch.dict('os.environ', {}):
                train_bounded_detector(a, options(a), run, a.weights)
            saved = json.loads((a.output/'run.json').read_text())
            self.assertEqual(saved['completedEpochs'], 1)
            self.assertEqual(saved['completionReason'], 'epoch_limit_reached')
            self.assertFalse(saved['convergenceClaim'])
            checkpoint = a.output/'train/weights/last.pt'
            verify_receipt(checkpoint, saved)
            _, metadata = run_metadata(a.output/'run.json', checkpoint, contract)
            self.assertEqual(metadata['architecture'], 'yolov8n-generic')


if __name__ == '__main__':
    unittest.main()
