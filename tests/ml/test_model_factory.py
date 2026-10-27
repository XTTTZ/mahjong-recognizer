"""Verify future model packages without GPU or untrusted checkpoint loading."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from ml.inference.manifest import FACE_TOKENS
from ml.inference.model import inspect_model


class ModelPackageTests(unittest.TestCase):
    def package(self, root, architecture, format, filenames):
        manifest = dict(architecture=architecture, format=format, version='test', resolution=960,
                        classes={str(i): v for i, v in enumerate(sorted(FACE_TOKENS))},
                        threshold=.25, acceptThreshold=.7, productionValidated=False,
                        classifierResolution=320, cropContext=.08)
        manifest['filesSha256'] = {}
        for filename in filenames:
            (root/filename).write_bytes(b'opaque-checkpoint')
            manifest['filesSha256'][filename] = hashlib.sha256(b'opaque-checkpoint').hexdigest()
        (root/'model.json').write_text(json.dumps(manifest))
        return manifest

    def test_both_cascade_stages_are_hashed_and_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.package(root, 'yolo-efficientnet-v2-s-cascade', 'two-stage-pt', ['detector.pt', 'classifier.pt'])
            self.assertFalse(inspect_model(root)['productionValidated'])
            (root/'classifier.pt').write_bytes(b'replaced')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                inspect_model(root)

    def test_yolo_package_rejects_corrupt_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.package(root, 'yolo26m', 'ultralytics-pt', ['weights.pt'])
            self.assertEqual(inspect_model(root)['architecture'], 'yolo26m')
            (root/'weights.pt').unlink()
            with self.assertRaises(ValueError):
                inspect_model(root)


if __name__ == '__main__':
    unittest.main()
