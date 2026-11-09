"""Variable research profiles cannot silently alter the frozen production profile."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ml.inference import candidate_onnx_runtime as target
from ml.inference.manifest import FACE_TOKENS, file_sha256
from ml.inference.onnx_runtime import detector_tiles, detector_tensor, verify_portable_directory
from infra.portable.candidate_export import base_manifest, source_contract


CLASSES = {str(i): name for i, name in enumerate(sorted(FACE_TOKENS))}


class CandidateProfiles(unittest.TestCase):
    def fixture(self, directory, resolution=640, size=192):
        detector = {'architecture': 'yolov8n-generic', 'resolution': resolution}
        classifier = {'architecture': 'mobilenet-v3-large', 'classes': CLASSES,
                      'config': {'resolution': size, 'context': .08}}
        manifest = base_manifest(detector, classifier, 'research-fixture')
        files = ('detector.onnx', 'classifier.onnx')
        for name in files:
            (directory / name).write_bytes(name.encode())
        manifest.update(filesSha256={n: file_sha256(directory/n) for n in files},
                        filesBytes={n: (directory/n).stat().st_size for n in files})
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        return manifest

    def test_resolution_and_hash_contracts_remain_research_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for resolution, count in [(640, 8400), (800, 13125), (960, 18900)]:
                manifest = self.fixture(root, resolution)
                self.assertEqual(target.head_points(resolution), count)
                self.assertEqual(target.verify_candidate_directory(root)[1], manifest)
                with self.assertRaises(ValueError):
                    verify_portable_directory(root)
            for key, value in [('threshold', .5), ('resolution', 1280), ('classifierResolution', 320),
                               ('researchOnly', False), ('cropContext', float('nan'))]:
                bad = dict(manifest); bad[key] = value
                (root/'manifest.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    target.verify_candidate_directory(root)
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'classifier.onnx').write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError, 'hash or size'):
                target.verify_candidate_directory(root)

    def test_dynamic_raw_shapes_reproduce_frozen960_geometry(self):
        image = Image.new('RGB', (789, 367))
        for size in (640, 800, 960):
            _, transform = detector_tensor(image, size)
            output = np.zeros((1, 5, target.head_points(size)), dtype=np.float32)
            output[0, :, 0] = [size/2, size/2, 80, 100, .9]
            output[0, :, 1] = [size/2+1, size/2+1, 80, 100, .8]
            tiles = target.candidate_detector_tiles(output, image.size, transform, size)
            self.assertEqual(len(tiles), 1)
            if size == 960:
                self.assertEqual(tiles, detector_tiles(output, image.size, transform))
            with self.assertRaisesRegex(ValueError, 'raw head'):
                target.candidate_detector_tiles(output[:, :, :-1], image.size, transform, size)
            output[0, 4, 0] = np.nan
            with self.assertRaisesRegex(ValueError, 'raw head'):
                target.candidate_detector_tiles(output, image.size, transform, size)

    def test_variable_classifier_batch_preserves_tail_and_all_detections(self):
        for size in (192, 224):
            for batch_size in (1, 4, 8):
                model = object.__new__(target.CandidateOnnxPredictor)
                model.manifest = {'resolution': 640, 'classifierResolution': size, 'cropContext': .08, 'classes': CLASSES}
                model.classifier_batch_size = batch_size
                model.detector = types.SimpleNamespace(run=lambda *args: [object()])
                seen = []
                def classify(_, feeds):
                    tensor = feeds['images']; seen.append(tensor.shape)
                    return [np.zeros((len(tensor), 42), dtype=np.float32)]
                model.classifier = types.SimpleNamespace(run=classify)
                detections = [{'bbox': [0, 0, 10, 10], 'score': .9} for _ in range(13)]
                with patch.object(target, 'candidate_detector_tiles', return_value=detections):
                    raw, timing = model.predict_with_timings(Image.new('RGB', (20, 20)))
                self.assertEqual(sum(x[0] for x in seen), 13)
                self.assertTrue(all(x[1:] == (3, size, size) for x in seen))
                self.assertEqual(len(raw), 13)
                self.assertEqual(timing['classifierBatches'], len(seen))
                with patch.object(target, 'candidate_detector_tiles', return_value=detections*16):
                    with self.assertRaisesRegex(ValueError, 'safety limit'):
                        model.predict_with_timings(Image.new('RGB', (20, 20)))

    def test_unsupported_192_efficientnet_is_rejected_before_framework_load(self):
        contract = {'ontology': 'faces42'}
        detector = {'architecture': 'yolov8n-generic', 'resolution': 640}
        classifier = {'architecture': 'efficientnet-v2-s', 'config': {'resolution': 192}}
        with patch('infra.portable.candidate_export.dataset_contract', return_value=contract), \
             patch('infra.portable.candidate_export.parent_contract', side_effect=[(detector, {}), (classifier, {})]):
            with self.assertRaisesRegex(ValueError, 'classifier'):
                source_contract('data', 'dr', 'dp', 'cr', 'cp')


if __name__ == '__main__':
    unittest.main()
