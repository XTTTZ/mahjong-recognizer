import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class FreezeTests(unittest.TestCase):
    def test_cannot_attach_another_experiments_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "weights.pth"
            checkpoint.write_bytes(b"checkpoint A")
            run = root / "run.json"
            run.write_text(json.dumps({"annotationSha256": "train-a", "architecture": "rfdetr-small", "resolution": 640}))
            report = root / "report.json"
            report.write_text(json.dumps({"checkpointSha256": hashlib.sha256(b"checkpoint B").hexdigest()}))
            result = subprocess.run([sys.executable, "-m", "ml.train.freeze", "--run", str(run),
                "--checkpoint", str(checkpoint), "--report", str(report), "--version", "test", "--out", str(root / "model")],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("checkpointSha256", result.stderr)
            self.assertFalse((root / "model").exists())

    def test_matching_artifacts_create_immutable_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "weights.pth"
            checkpoint.write_bytes(b"checkpoint A")
            run = root / "run.json"
            run.write_text(json.dumps({"annotationSha256": "train-a", "architecture": "rfdetr-small", "resolution": 640,
                                       "classes": {str(i): c["token"] for i, c in enumerate(json.loads(Path("docs/data/class-map.json").read_text())["classes"])}, "codeCommit": "test",
                                       "resumeSignature": {"fixture": "checked-not-real-model"}, "validationAnnotationSha256": "valid-a"}))
            report = root / "report.json"
            report.write_text(json.dumps({"checkpointSha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "runSha256": hashlib.sha256(run.read_bytes()).hexdigest(), "trainAnnotationSha256": "train-a",
                "architecture": "rfdetr-small", "resolution": 640, "split": "validation_only", "bestThreshold": .5,
                "validationAnnotationSha256": "valid-a", "resumeSignatureSha256": hashlib.sha256(json.dumps({"fixture": "checked-not-real-model"}, sort_keys=True).encode()).hexdigest()}))
            arguments = [sys.executable, "-m", "ml.train.freeze", "--run", str(run), "--checkpoint", str(checkpoint),
                         "--report", str(report), "--version", "test", "--out", str(root / "model")]
            result = subprocess.run(arguments, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "model/model.json").read_text())
            self.assertFalse(manifest["productionValidated"])
            self.assertEqual(manifest["threshold"], .5)
            self.assertNotEqual(subprocess.run(arguments, capture_output=True).returncode, 0)
