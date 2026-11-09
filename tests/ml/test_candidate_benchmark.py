"""Portable quality/timing evidence with inert predictors, never model inference."""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from infra.portable import candidate_benchmark as target
from infra.portable.quality_gate import paired_quality_gate
from ml.inference.manifest import FACE_TOKENS

CLASSES = {str(i): token for i, token in enumerate(sorted(FACE_TOKENS))}


def frozen_fixture():
    payload, images, annotations = [], [], []
    for ident in range(1, 151):
        pixels = BytesIO(); Image.new('RGB', (200, 20), (ident, 0, 0)).save(pixels, format='PNG')
        data = pixels.getvalue()
        payload.append({'id': ident, 'data': data})
        images.append({'id': ident, 'file_name': f'{ident}.png', 'sha256': hashlib.sha256(data).hexdigest()})
        # Frozen primary82 and13/14 subset45, as required by the report contract.
        count = 13 if ident <= 45 else 10 if ident <= 82 else 1
        for tile in range(count):
            annotations.append({'image_id': ident, 'category_id': 1, 'bbox': [tile*12, 0, 10, 15]})
    return payload, {'images': images, 'annotations': annotations}


def fake_predictors(annotation, *, repeat_failure=None, wrong_image=None):
    truths = defaultdict(list)
    for a in annotation['annotations']:
        truths[a['image_id']].append({'classId': CLASSES['0'], 'bbox': a['bbox'], 'score': .9})
    def factory(kind):
        class Fake:
            def __init__(self, *_):
                self.calls = defaultdict(int)
                self.manifest = {'classes': CLASSES, 'filesBytes': {'detector.onnx': 100, 'classifier.onnx': 200},
                                 'filesSha256': {'detector.onnx': 'a'*64, 'classifier.onnx': 'b'*64}}
            def predict_with_timings(self, image):
                ident = image.getpixel((0, 0))[0]
                self.calls[ident] += 1
                if kind == 'candidate' and ident == repeat_failure and self.calls[ident] == 2:
                    raise ValueError('inert repeated-call failure')
                raw = [dict(tile) for tile in truths[ident]]
                if kind == 'candidate' and ident == wrong_image: raw[0]['classId'] = CLASSES['1']
                return raw, {'totalMs': 100 if kind == 'candidate' else 200}
        return Fake
    return factory('baseline'), factory('candidate')


class BenchmarkEvidenceTests(unittest.TestCase):
    def run_fixture(self, **behavior):
        payload, annotation = frozen_fixture()
        baseline, candidate = fake_predictors(annotation, **behavior)
        with patch('ml.inference.onnx_runtime.OnnxCascadePredictor', baseline), \
             patch('ml.inference.candidate_onnx_runtime.CandidateOnnxPredictor', candidate), \
             patch.dict(sys.modules, {'onnxruntime': SimpleNamespace(__version__='mock-no-inference')}), \
             redirect_stdout(StringIO()):
            return target.benchmark('baseline', 'candidate', payload, annotation)

    def test_complete_paired_quality_and_timing_success_has_exact_denominators(self):
        report = self.run_fixture()
        self.assertTrue(report['qualityGate']['passed'])
        self.assertTrue(report['warmLatencyTargetPassed'])
        self.assertFalse(report['productionPromotion']); self.assertFalse(report['phoneHardwareTested'])
        self.assertEqual(report['summaries']['candidate']['all150']['successes'], 150)
        self.assertEqual(report['summaries']['candidate']['primary82']['images'], 82)
        speeds = report['summaries']['candidate']['thirteenFourteen45']
        self.assertEqual((speeds['images'], speeds['timedImages']), (45, 45))
        self.assertGreaterEqual(speeds['p95Ms'], 100)
        self.assertIn('normalizationMs', report['timingRuns']['candidate'][0]['timings'])
        self.assertEqual(len(report['timingRuns']['candidate']), 300)

    def test_new_wrong_accept_fails_gate_even_with_fast_runtime(self):
        report = self.run_fixture(wrong_image=83)
        self.assertFalse(report['qualityGate']['passed'])
        self.assertEqual(report['qualityGate']['newWrongAcceptIds'], [83])
        self.assertEqual(report['summaries']['candidate']['all150']['wrongAccepts'], 1)
        self.assertEqual(report['summaries']['candidate']['all150']['successes'], 149)

    def test_repeat_failure_is_not_model_exact_and_stays_in_quality_denominator(self):
        report = self.run_fixture(repeat_failure=3)
        self.assertFalse(report['qualityGate']['passed']); self.assertFalse(report['warmLatencyTargetPassed'])
        self.assertEqual(report['qualityGate']['candidateFailedIds'], [3])
        metric = report['perImage']['candidate'][2]['metrics']
        self.assertTrue(metric['failed']); self.assertFalse(metric['hand_exact_match'])
        self.assertFalse(metric['model_exact']); self.assertFalse(metric['multiset_exact'])
        total = report['summaries']['candidate']['all150']
        self.assertEqual((total['images'], total['successes'], total['failed_requests']), (150, 149, 1))
        self.assertEqual(total['model_exact'], 149/150)
        self.assertEqual(report['summaries']['candidate']['thirteenFourteen45']['timedImages'], 44)

    def test_default_main_checks_only_never_instantiates_models_or_creates_report(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'never-created.json'
            args = argparse.Namespace(baseline_dir=Path('baseline'), candidate_dir=Path('candidate'),
                valid_dir=Path('valid'), out=out, threads=2, batch=4, repeats=2, run_local=False, run_cloud=False)
            with patch.object(argparse.ArgumentParser, 'parse_args', return_value=args), \
                 patch('ml.inference.onnx_runtime.verify_portable_directory'), \
                 patch('ml.inference.candidate_onnx_runtime.verify_candidate_directory'), \
                 patch.object(target, 'validation_payload', return_value=frozen_fixture()), \
                 patch.object(target, 'benchmark', side_effect=AssertionError('must not execute')), \
                 patch.dict(sys.modules, {'modal': None, 'onnxruntime': None}), redirect_stdout(StringIO()) as output:
                target.main()
            self.assertFalse(out.exists())
            receipt = json.loads(output.getvalue()); self.assertFalse(receipt['inferenceStarted'])
            self.assertFalse(receipt['cloudStarted']); self.assertEqual(receipt['images'], 150)

    def test_validation_rejects_changed_pixels_and_non_valid_directory(self):
        payload, annotation = frozen_fixture()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'valid'; root.mkdir()
            encoded = json.dumps(annotation).encode(); (root/'_annotations.coco.json').write_bytes(encoded)
            for item in payload: (root/f"{item['id']}.png").write_bytes(item['data'])
            with patch.object(target, 'ANNOTATION_SHA', hashlib.sha256(encoded).hexdigest()):
                self.assertEqual(len(target.validation_payload(root)[0]), 150)
                (root/'1.png').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'pixels'): target.validation_payload(root)
            with self.assertRaisesRegex(ValueError, 'valid split'): target.validation_payload(root.parent/'test')

    def test_quality_gate_rejects_mismatched_ids_duplicates_and_failed_reference(self):
        metric = {'truth_count': 1, 'hand_exact_match': True, 'model_exact': True, 'accepted': True, 'failed': False}
        old = [{'imageId': 1, 'metrics': metric}]
        with self.assertRaises(ValueError): paired_quality_gate(old, [{'imageId': 2, 'metrics': metric}])
        with self.assertRaises(ValueError): paired_quality_gate(old+old, old)
        failed = [{'imageId': 1, 'metrics': {**metric, 'failed': True}}]
        self.assertFalse(paired_quality_gate(failed, old)['passed'])


if __name__ == '__main__':
    unittest.main()
