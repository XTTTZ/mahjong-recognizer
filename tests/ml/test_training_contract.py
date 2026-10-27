import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ml.eval.metrics import Tile
from ml.eval.operating_point import score_predictions
from ml.inference.model import inspect_model
from ml.train.provenance import check_evaluation_data, check_resume, dataset_contract
from ml.train.rfdetr_train import checkpoint_contract

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = [{"id": c["coco_id"], "name": c["token"]} for c in json.loads(
    (ROOT / "docs/data/class-map.json").read_text())["classes"]]


def data_fixture(root):
    for split in ("train", "valid"):
        directory = root / split
        directory.mkdir()
        (directory / "_annotations.coco.json").write_text(json.dumps({"categories": CATEGORIES,
            "images": [{"id": 0, "file_name": "image.jpg", "width": 100, "height": 80, "group_id": split}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 2, "bbox": [0, 0, 10, 20]}]}))
    # Deliberately invalid: training metadata and preflight must never open it.
    (root / "test").mkdir()
    (root / "test/_annotations.coco.json").write_text("sealed test must not be opened")


class TrainingContractTests(unittest.TestCase):
    def test_cpu_preflight_does_not_import_torch_open_test_or_create_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_fixture(root)
            result = subprocess.run([sys.executable, "-m", "ml.train.rfdetr_train", "--dataset", str(root),
                "--output", str(root / "new-run"), "--preflight-only"], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)["contract"]["testOpened"])
            self.assertFalse((root / "new-run").exists())
            result = subprocess.run([sys.executable, "-c", "import sys; import ml.train.rfdetr_train; assert 'torch' not in sys.modules"], cwd=ROOT, capture_output=True)
            self.assertEqual(result.returncode, 0)

    def test_validation_pixel_replacement_cannot_be_paired_with_original_run(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_fixture(root)
            for split in ("train", "valid"):
                Image.new("RGB", (100, 80), "white" if split == "valid" else "gray").save(root / split / "image.jpg", format="PNG")
            original = dataset_contract(root, verify_images=True)
            run = {"classes": original["classes"], "resumeSignature": {"data": original["splits"]},
                   "annotationSha256": original["splits"]["train"]["annotationSha256"],
                   "validationAnnotationSha256": original["splits"]["valid"]["annotationSha256"]}
            check_evaluation_data(run, original)
            Image.new("RGB", (100, 80), "black").save(root / "valid/image.jpg")
            changed = dataset_contract(root, verify_images=True)
            self.assertEqual(original["splits"]["valid"]["annotationSha256"], changed["splits"]["valid"]["annotationSha256"])
            with self.assertRaisesRegex(ValueError, "pixels"):
                check_evaluation_data(run, changed)

    def test_faces42_schema_and_exact_image_leakage_guard(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_fixture(root)
            for split in ("train", "valid"):
                annotation = root / split / "_annotations.coco.json"
                data = json.loads(annotation.read_text())
                data["categories"] = [c for c in data["categories"] if c["name"] != "back"]
                annotation.write_text(json.dumps(data))
                Image.new("RGB", (100, 80), "white").save(root / split / "image.jpg")
            self.assertEqual(dataset_contract(root)["ontology"], "faces42")
            with self.assertRaisesRegex(ValueError, "byte leakage"):
                dataset_contract(root, verify_images=True)

    def test_group_leakage_and_out_of_bounds_truth_stop_before_gpu(self):
        for mutation in ("group", "bbox"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data_fixture(root)
                path = root / "valid/_annotations.coco.json"
                data = json.loads(path.read_text())
                if mutation == "group":
                    data["images"][0]["group_id"] = "train"
                else:
                    data["annotations"][0]["bbox"] = [90, 0, 20, 20]
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    dataset_contract(root)

    def test_changed_data_and_foreign_checkpoint_cannot_resume(self):
        signature = {"train": "clean-v3", "valid": "clean-v3-valid"}
        with self.assertRaisesRegex(ValueError, "identical data"):
            check_resume({"resumeSignature": {"train": "mixed-v2"}}, signature, "run/a.ckpt", "run")
        with self.assertRaisesRegex(ValueError, "another or legacy"):
            checkpoint_contract({"epoch": 0, "optimizer_states": [{}], "lr_schedulers": [{}]}, signature)
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        checkpoint = {"args": {"notes": {"mahjongResumeSignature": digest}}, "epoch": 0,
                      "optimizer_states": [{"state": {}}], "lr_schedulers": [{}]}
        checkpoint_contract(checkpoint, signature)
        del checkpoint["optimizer_states"]
        with self.assertRaisesRegex(ValueError, "optimizer"):
            checkpoint_contract(checkpoint, signature)


class APIEvaluationParityTests(unittest.TestCase):
    def test_clipping_and_rejection_match_the_served_result(self):
        truth = [Tile("1m", (0, 0, 10, 20))]
        raw = [{"classId": "1m", "bbox": [-10, 0, 20, 20], "score": .9}]
        self.assertTrue(score_predictions(truth, raw, 100, 80, .25, .7)["hand_exact_match"])
        raw[0]["score"] = .6
        result = score_predictions(truth, raw, 100, 80, .25, .7)
        self.assertTrue(result["model_exact"])
        self.assertFalse(result["hand_exact_match"])

    def test_score_equal_to_detector_threshold_is_excluded(self):
        truth = [Tile("1m", (0, 0, 10, 20))]
        raw = [{"classId": "1m", "bbox": [0, 0, 10, 20], "score": .5}]
        result = score_predictions(truth, raw, 100, 80, .5, .7)
        self.assertEqual(result["prediction_count"], 0)
        self.assertFalse(result["hand_exact_match"])

    def test_http_crowded_scene_failure_remains_a_failed_image(self):
        raw = [{"classId": "1m", "bbox": [0, 0, 10, 20], "score": .9}] * 201
        result = score_predictions([], raw, 100, 80, .25, .7)
        self.assertTrue(result["failed"])
        self.assertFalse(result["hand_exact_match"])


class OfflineManifestTests(unittest.TestCase):
    def test_integrity_validation_runs_without_loading_a_gpu_framework(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "checkpoint.pth"
            weights.write_bytes(b"hash contract fixture")
            manifest = {"architecture": "rfdetr-small", "checkpoint": weights.name,
                "checkpointSha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "classes": {str(i): c["name"] for i, c in enumerate(CATEGORIES)},
                "version": "fixture-v1", "resolution": 640, "threshold": .35, "acceptThreshold": .7}
            (root / "model.json").write_text(json.dumps(manifest))
            self.assertEqual(inspect_model(root)["version"], "fixture-v1")
            weights.write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                inspect_model(root)
            manifest["checkpoint"] = "../foreign.pth"
            (root / "model.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "model-local"):
                inspect_model(root)
