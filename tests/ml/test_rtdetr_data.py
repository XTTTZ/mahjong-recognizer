"""Pure RT-DETR data/manifest contracts: Pillow is required; torch is not."""
import json
import math
import random
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageOps

from ml.inference.rtdetr_model import file_sha256, prepare_input, restore_tiles, verify_model_directory
from ml.train.rtdetr_train import CocoHands, append_metrics, oriented_bbox, photometric, runtime_versions, shared_code_hashes

ROOT = Path(__file__).resolve().parents[2]


def categories():
    classes = json.loads((ROOT / "docs/data/class-map.json").read_text(encoding="utf-8"))["classes"]
    return [{"id": item["coco_id"], "name": item["token"]} for item in classes]


def coco_fixture(directory: Path, *, orientation=1):
    image = Image.new("RGB", (64, 48), "white")
    image.getexif()[274] = orientation
    image.save(directory / "a.png", exif=image.getexif())
    coco = {"categories": categories(),
            "images": [{"id": 7, "file_name": "a.png", "width": 64, "height": 48, "group_id": "group-a"}],
            "annotations": [{"id": 1, "image_id": 7, "category_id": 43, "bbox": [8, 10, 20, 25]}]}
    (directory / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    return coco


class EXIFBoxTests(unittest.TestCase):
    def test_all_eight_exif_orientations_preserve_pixel_box_alignment(self):
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                image = Image.new("RGB", (50, 30), "white")
                ImageDraw.Draw(image).rectangle((10, 5, 19, 12), fill="black")
                image.getexif()[274] = orientation
                pixel_box = ImageOps.invert(ImageOps.exif_transpose(image)).getbbox()
                expected = [pixel_box[0], pixel_box[1], pixel_box[2] - pixel_box[0], pixel_box[3] - pixel_box[1]]
                self.assertEqual(oriented_bbox([10, 5, 10, 8], 50, 30, orientation), expected)

    def test_invalid_truth_is_rejected_instead_of_silently_clipped(self):
        for box in ([0, 0, 0, 5], [0, 0, 5, -1], [-1, 0, 5, 5], [49, 0, 5, 5],
                    [0, 0, math.nan, 5], [0, 0, 5, math.inf]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                oriented_bbox(box, 50, 30, 1)


class CocoMappingTests(unittest.TestCase):
    def test_face_only_42_category_head_does_not_keep_an_untrained_back_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coco = coco_fixture(root)
            coco["categories"] = [c for c in coco["categories"] if c["name"] != "back"]
            coco["annotations"][0]["category_id"] = 42
            (root / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
            dataset = CocoHands(root)
            _, target, _ = dataset[0]
            self.assertEqual(len(dataset.classes), 42)
            self.assertNotIn("back", dataset.classes.values())
            self.assertEqual(target["annotations"][0]["category_id"], 41)

    def test_original_category_43_becomes_model_index_42(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coco_fixture(root)
            dataset = CocoHands(root)
            image, target, record = dataset[0]
            self.assertEqual(image.size, (64, 48))
            self.assertEqual(len(dataset.classes), 43)
            self.assertEqual(dataset.class_ids, {i: i - 1 for i in range(1, 44)})
            self.assertEqual(target["annotations"][0]["category_id"], 42)
            self.assertEqual(target["annotations"][0]["bbox"], [8, 10, 20, 25])
            self.assertEqual(record["group_id"], "group-a")

    def test_rotated_encoded_image_and_truth_use_the_same_orientation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coco_fixture(root, orientation=6)
            image, target, _ = CocoHands(root)[0]
            self.assertEqual(image.size, (48, 64))
            self.assertEqual(target["annotations"][0]["bbox"], [13, 8, 25, 20])

    def test_parent_category_cannot_be_mistaken_for_a_mahjong_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coco = coco_fixture(root)
            coco["categories"].insert(0, {"id": 0, "name": "mahjong-yolo"})
            (root / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "43-class"):
                CocoHands(root)

    def test_mismatched_metadata_dimensions_do_not_silently_shift_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coco = coco_fixture(root)
            coco["images"][0]["width"] = 63
            (root / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dimensions"):
                CocoHands(root)[0]

    def test_unknown_annotation_category_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coco = coco_fixture(root)
            coco["annotations"][0]["category_id"] = 44
            (root / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown"):
                CocoHands(root)


class FrozenRTDETRTests(unittest.TestCase):
    def manifest_fixture(self, root):
        files = ("config.json", "preprocessor_config.json", "model.safetensors")
        for filename in files:
            (root / filename).write_bytes(b"metadata-integrity-fixture")
        manifest = {"architecture": "rtdetr-v2-r18vd", "format": "huggingface-pretrained",
                    "classes": {str(i): category["name"] for i, category in enumerate(categories())},
                    "threshold": .35, "acceptThreshold": .7, "version": "fixture", "resolution": 640,
                    "filesSha256": {filename: file_sha256(root / filename) for filename in files}}
        (root / "model.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_hash_guard_covers_processor_configuration_and_weights(self):
        for filename in ("config.json", "preprocessor_config.json", "model.safetensors"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.manifest_fixture(root)
                verify_model_directory(root)
                (root / filename).write_bytes(b"altered")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    verify_model_directory(root)

    def test_manifest_cannot_omit_the_processor_from_integrity_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest_fixture(root)
            del manifest["filesSha256"]["preprocessor_config.json"]
            (root / "model.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest"):
                verify_model_directory(root)

    def test_module_import_and_cli_metadata_do_not_import_torch_or_transformers(self):
        script = "import sys; import ml.train.rtdetr_train, ml.inference.rtdetr_model; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class PreprocessingTests(unittest.TestCase):
    def test_resume_signature_covers_common_validation_code_and_runtime_versions(self):
        import hashlib
        expected = ("eval/metrics.py", "eval/operating_point.py", "inference/postprocess.py",
                    "inference/manifest.py", "train/provenance.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in expected:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"first-validation-definition")
            before = shared_code_hashes(root)
            (root / "eval/operating_point.py").write_bytes(b"changed-threshold-rule")
            after = shared_code_hashes(root)
            self.assertNotEqual(before, after)
            self.assertEqual(after["eval/operating_point.py"], hashlib.sha256(b"changed-threshold-rule").hexdigest())
        with patch("ml.train.rtdetr_train.importlib.metadata.version", side_effect=lambda name: "pinned-" + name):
            versions = runtime_versions("2.8.0+cu128")
            self.assertEqual(versions["torch"], "2.8.0+cu128")
            self.assertEqual(versions["faster-coco-eval"], "pinned-faster-coco-eval")
            self.assertIn("Pillow", versions)

    def test_letterbox_boxes_roundtrip_and_original_target_is_unchanged(self):
        image = Image.new("RGB", (101, 37), "white")
        target = {"image_id": 1, "annotations": [{"id": 1, "category_id": 0, "bbox": [10, 5, 20, 20], "area": 400}]}
        original = json.dumps(target, sort_keys=True)
        prepared, transformed, geometry = prepare_input(image, target, resolution=640, mode="letterbox")
        self.assertEqual(prepared.size, (640, 640))
        self.assertEqual(prepared.getpixel((0, 0)), (114, 114, 114))
        self.assertEqual(prepared.getpixel((320, 320)), (255, 255, 255))
        restored = restore_tiles([{"classId": "1m", "bbox": transformed["annotations"][0]["bbox"], "score": .9}], geometry)
        for actual, expected in zip(restored[0]["bbox"], target["annotations"][0]["bbox"]):
            self.assertAlmostEqual(actual, expected, places=10)
        self.assertEqual(json.dumps(target, sort_keys=True), original)

    def test_warp_path_does_not_preempt_the_saved_processor(self):
        image = Image.new("RGB", (100, 40), "white")
        target = {"image_id": 1, "annotations": []}
        prepared, transformed, geometry = prepare_input(image, target, resolution=640, mode="warp")
        self.assertIs(prepared, image)
        self.assertIs(transformed, target)
        self.assertIsNone(geometry)

    def test_photometric_is_seeded_and_does_not_flip_or_change_canvas(self):
        image = Image.new("RGB", (100, 50), "white")
        ImageDraw.Draw(image).rectangle((0, 0, 45, 49), fill=(20, 30, 200))
        first = photometric(image, random.Random(71))
        second = photometric(image, random.Random(71))
        self.assertEqual(first.size, image.size)
        self.assertEqual(first.tobytes(), second.tobytes())
        left, right = first.getpixel((10, 25)), first.getpixel((90, 25))
        self.assertGreater(left[2], left[0] * 2)
        self.assertLess(max(right) - min(right), 8)
        self.assertEqual(image.getpixel((10, 25)), (20, 30, 200))

    def test_metrics_do_not_reuse_previous_validation_for_unvalidated_epoch(self):
        import csv
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.csv"
            chosen = {"hand_exact_match": .5, "model_exact": .6, "accepted_coverage": .8}
            append_metrics(path, 1, 10, 8.5, chosen, {"mAP50": .7, "mAP50_95": .4}, 100, .0001)
            append_metrics(path, 2, 20, 7.2, None, None, 200, .00009)
            with path.open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[0]["epoch"], "0")
            self.assertEqual(rows[1]["val/mAP_50_95"], "")
            self.assertEqual(rows[1]["train/loss"], "7.2")


if __name__ == "__main__":
    unittest.main()
