"""CPU contracts for YOLO derivatives and actual-box cascade preprocessing."""
import json
import contextlib
import io
from pathlib import Path
import random
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from ml.data.prepare_yolo import normalized_box, prepare as prepare_yolo, verify_prepared as verify_yolo
from ml.data.prepare_crops import prepare as prepare_crops, verify_prepared as verify_crops
from ml.inference.cascade_model import combine_tiles, crop_face, verify_model_directory as verify_cascade
from ml.inference.yolo_model import verify_model_directory as verify_model
from ml.inference.manifest import FACE_TOKENS
from ml.train.provenance import dataset_contract, sha256


def fixture(root):
    classes = {str(index): token for index, token in enumerate(sorted(FACE_TOKENS))}
    for index, split in enumerate(("train", "valid")):
        directory = root / split
        directory.mkdir(parents=True)
        Image.new("RGB", (80, 100), "white" if index == 0 else "green").save(directory / "source.png")
        coco = {"categories": [{"id": int(i)+1, "name": token} for i, token in classes.items()],
            "images": [{"id": 1, "file_name": "source.png", "width": 80, "height": 100, "group_id": split}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 42, "bbox": [10, 20, 30, 40]}]}
        (directory / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    # Any accidental read of test is invalid JSON; correct code never touches it.
    (root / "test").mkdir()
    (root / "test/_annotations.coco.json").write_text("SEALED_DO_NOT_READ", encoding="utf-8")
    return classes


class DerivativeTests(unittest.TestCase):
    def test_yolo_preserves_pixels_and_explicit_zero_based_face_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "source")
            metadata = prepare_yolo(root / "source", root / "yolo")
            self.assertEqual((root / "yolo/labels/train/1.txt").read_text().split()[0], "41")
            self.assertEqual(sha256(root / "source/train/source.png"), sha256(root / "yolo/images/train/1.png"))
            self.assertFalse(metadata["testOpened"])
            verify_yolo(root / "yolo", dataset_contract(root / "source", True))
            self.assertEqual(normalized_box([10, 20, 30, 40], 80, 100), [.3125, .4, .375, .4])

    def test_generic_detector_has_exactly_one_class_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "source")
            before = sha256(root / "source/train/_annotations.coco.json")
            metadata = prepare_yolo(root / "source", root / "yolo", generic=True)
            self.assertEqual(metadata["names"], ["tile"])
            self.assertEqual((root / "yolo/labels/train/1.txt").read_text().split()[0], "0")
            self.assertEqual(before, sha256(root / "source/train/_annotations.coco.json"))

    def test_tampered_labels_and_pixels_are_rejected(self):
        for component in ("labels/train/1.txt", "images/train/1.png"):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture(root / "source")
                contract = dataset_contract(root / "source", True)
                prepare_yolo(root / "source", root / "yolo")
                (root / "yolo" / component).write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    verify_yolo(root / "yolo", contract)

    def test_preflight_writes_nothing_and_reports_unprepared_derivative(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "source")
            for prepare in (prepare_yolo, prepare_crops):
                report = prepare(root / "source", root / "missing", preflight_only=True)
                self.assertFalse(report["derivativeReady"])
                self.assertFalse((root / "missing").exists())

    def test_crop_indices_preserve_split_groups_and_reject_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "source")
            contract = dataset_contract(root / "source", True)
            prepare_crops(root / "source", root / "crops")
            train = json.loads((root / "crops/train.json").read_text())["records"][0]
            valid = json.loads((root / "crops/valid.json").read_text())["records"][0]
            self.assertNotEqual(train["groupId"], valid["groupId"])
            self.assertEqual(train["label"], 41)
            verify_crops(root / "crops", contract)
            (root / "crops/train.json").write_text("{}")
            with self.assertRaises(ValueError):
                verify_crops(root / "crops", contract)

    def test_source_group_leakage_is_rejected_before_any_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "source")
            path = root / "source/valid/_annotations.coco.json"
            data = json.loads(path.read_text())
            data["images"][0]["group_id"] = "train"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "leakage"):
                prepare_crops(root / "source", root / "crops")
            self.assertFalse((root / "crops").exists())


class CascadeGeometryTests(unittest.TestCase):
    def test_full_crop_is_padded_without_central_crop_or_aspect_distortion(self):
        image = Image.new("RGB", (30, 60), "white")
        image.putpixel((0, 0), (255, 0, 0))
        image.putpixel((29, 59), (0, 0, 255))
        crop = crop_face(image, [0, 0, 30, 60], resolution=60, context=0)
        self.assertEqual(crop.size, (60, 60))
        self.assertEqual(crop.getpixel((15, 0)), (255, 0, 0))
        self.assertEqual(crop.getpixel((44, 59)), (0, 0, 255))
        self.assertEqual(crop.getpixel((0, 0)), (114, 114, 114))

    def test_jitter_is_seeded_and_context_box_outside_image_is_rejected(self):
        image = Image.new("RGB", (80, 100), "white")
        a = crop_face(image, [1, 2, 20, 40], rng=random.Random(7))
        b = crop_face(image, [1, 2, 20, 40], rng=random.Random(7))
        self.assertEqual(a.tobytes(), b.tobytes())
        with self.assertRaises(ValueError):
            crop_face(image, [500, 500, 10, 10])

    def test_classifier_probabilities_are_paired_with_actual_detector_boxes(self):
        classes = {"0": "1m", "1": "2m"}
        detector = [{"bbox": [5, 6, 20, 30], "classId": "tile", "score": .8}]
        result = combine_tiles(detector, [[.1, .9]], classes)
        self.assertEqual(result[0]["bbox"], [5, 6, 20, 30])
        self.assertEqual(result[0]["classId"], "2m")
        self.assertAlmostEqual(result[0]["score"], .72)
        with self.assertRaises(ValueError):
            combine_tiles(detector, [], classes)


class ManifestTests(unittest.TestCase):
    def test_pinned_yolo_oom_policy_cannot_silently_change_batch(self):
        from ml.train.yolo_train import enforce_fixed_batch
        trainer = SimpleNamespace(batch_size=8, args=SimpleNamespace(batch=8), _oom_retries=0)
        enforce_fixed_batch(trainer, 8)
        self.assertEqual(trainer._oom_retries, 3)
        trainer.batch_size = 4
        with self.assertRaisesRegex(RuntimeError, "batch"):
            enforce_fixed_batch(trainer, 8)
    def test_cascade_binds_both_stages_and_crop_geometry_without_importing_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("detector.pt", "classifier.pt"):
                (root / name).write_bytes(name.encode())
            classes = {str(index): token for index, token in enumerate(sorted(FACE_TOKENS))}
            manifest = {"architecture": "yolo-efficientnet-v2-s-cascade", "format": "two-stage-pt",
                "classes": classes, "version": "test-v1", "resolution": 960, "threshold": .25,
                "acceptThreshold": .7, "classifierResolution": 224, "cropContext": .08,
                "filesSha256": {name: sha256(root / name) for name in ("detector.pt", "classifier.pt")}}
            (root / "model.json").write_text(json.dumps(manifest))
            verify_cascade(root)
            (root / "classifier.pt").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "SHA"):
                verify_cascade(root)

    def test_yolo_manifest_rejects_wrong_family_or_tampered_weight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "weights.pt").write_bytes(b"test")
            manifest = {"architecture": "yolo26m", "format": "ultralytics-pt", "version": "test-v1",
                "classes": {str(index): token for index, token in enumerate(sorted(FACE_TOKENS))},
                "resolution": 960, "threshold": .25, "acceptThreshold": .7,
                "filesSha256": {"weights.pt": sha256(root / "weights.pt")}}
            (root / "model.json").write_text(json.dumps(manifest))
            verify_model(root)
            (root / "weights.pt").write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "weights"):
                verify_model(root)

    def test_fresh_queue_log_directory_is_accepted_in_a_real_child_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("status.json", "stdout.log", "stderr.log"):
                (root / name).write_text("{}")
            code = "from ml.train.yolo_train import admit_output;import sys;admit_output(sys.argv[1])"
            accepted = subprocess.run([sys.executable, "-c", code, str(root)], capture_output=True)
            self.assertEqual(accepted.returncode, 0, accepted.stderr.decode())
            (root / "run.json").write_text("{}")
            rejected = subprocess.run([sys.executable, "-c", code, str(root)], capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)

    def test_admission_rejects_checkpoint_or_subdirectory_inside_old_run(self):
        from ml.train.yolo_train import admit_output
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "last.pt").write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                admit_output(root)


class FullImageEvaluationTests(unittest.TestCase):
    def test_correct_glyph_with_wrong_detector_box_is_not_a_complete_hand_success(self):
        from ml.eval.yolo_evaluate import evaluate
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classes = fixture(root / "source")
            class Predictor:
                def predict(self, image):
                    return [{"classId": classes["41"], "bbox": [60, 60, 10, 10], "score": .99}]
            with patch("ml.eval.coco_metrics.coco_metrics", return_value={"mAP50_95": 0.}), contextlib.redirect_stdout(io.StringIO()):
                report = evaluate(Predictor(), root / "source", root / "eval",
                    {"dataContract": dataset_contract(root / "source", True)})
            metric = report["thresholds"][str(report["bestThreshold"])]["all"]
            self.assertEqual(metric["multiset_exact"], 1.)
            self.assertEqual(metric["hand_exact_match"], 0.)
            self.assertEqual(metric["model_exact"], 0.)

    def test_prediction_failures_remain_in_denominator(self):
        from ml.eval.yolo_evaluate import evaluate
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "source")
            class Predictor:
                def predict(self, image):
                    raise RuntimeError("classifier failure")
            with patch("ml.eval.coco_metrics.coco_metrics", return_value={"mAP50_95": 0.}), contextlib.redirect_stdout(io.StringIO()):
                report = evaluate(Predictor(), root / "source", root / "eval",
                    {"dataContract": dataset_contract(root / "source", True)})
            metric = report["thresholds"][str(report["bestThreshold"])]["all"]
            self.assertEqual(metric["images"], 1)
            self.assertEqual(metric["failed_requests"], 1)
            self.assertEqual(report["inferenceFailedImages"], 1)


class CandidateSelectionTests(unittest.TestCase):
    def test_byte_identical_best_and_last_are_only_evaluated_once(self):
        from ml.eval.evaluate_yolo_candidates import candidates
        from ml.train.yolo_train import checkpoint_receipt
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classes = {str(index): token for index, token in enumerate(sorted(FACE_TOKENS))}
            signature = {"data": {"train": "frozen", "valid": "frozen"}}
            run = {"finishedAt": "2026-10-26", "resumeSignature": signature, "classes": classes}
            (root / "run.json").write_text(json.dumps(run))
            for name in ("best.pt", "last.pt"):
                (root / name).write_bytes(b"identical model bytes")
                checkpoint_receipt(root / name, signature, classes)
            self.assertEqual(len(candidates(root / "best.pt", root / "run.json")), 1)

    def test_candidate_ranking_prefers_complete_hands_over_higher_box_ap(self):
        from ml.eval.evaluate_yolo_candidates import ranking
        def report(hand, box_ap):
            return {"selectionGroup": "raw_ten_or_more_tiles", "bestThreshold": .25, "cocoAP": {"mAP50_95": box_ap},
                "thresholds": {"0.25": {"raw_ten_or_more_tiles": {"hand_exact_match": hand, "model_exact": hand, "accepted_coverage": 1.}}}}
        self.assertGreater(ranking(report(.8, .5)), ranking(report(.5, .9)))


if __name__ == "__main__":
    unittest.main()
