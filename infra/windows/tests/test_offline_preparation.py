from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile

from infra.windows.offline_bundle import MANIFEST, build, digest, extract_new, inspect_zip, safe_name, verify

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("prepare_windows_data", ROOT / "scripts/prepare_windows_data.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


class OfflinePreparationTests(unittest.TestCase):
    def fixture(self, directory: Path):
        archive, manifest, metadata = directory / "raw.zip", directory / "splits.jsonl", directory / "metadata.json"
        rows, expected = [], {}
        with zipfile.ZipFile(archive, "w") as zipped:
            for index, split in enumerate(("train", "valid", "test")):
                uid = f"{split}/source-{index}.jpg"
                data = f"source-{index}".encode()
                rows.append({"source_id": uid, "sha256": hashlib.sha256(data).hexdigest(), "group_id": f"g{index}", "split": split, "exclusion_reason": None})
                zipped.writestr(uid, data)
                source = {"images": [{"id": 0, "file_name": f"source-{index}.jpg", "width": 4, "height": 3}],
                          "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 2, 2], "iscrowd": 0}],
                          "categories": [{"id": 1, "name": "1m"}]}
                zipped.writestr(f"{split}/_annotations.coco.json", json.dumps(source))
                prepared = {"images": [{"id": 0, "file_name": hashlib.sha256(uid.encode()).hexdigest()[:16] + ".jpg", "width": 4, "height": 3, "source_id": uid, "group_id": f"g{index}"}],
                            "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 0}],
                            "categories": [{"id": 1, "name": "1m", "supercategory": "mahjong"}]}
                expected[split] = {"images": 1, "annotations": 1, "annotation_sha256": hashlib.sha256(json.dumps(prepared, ensure_ascii=False).encode()).hexdigest()}
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        value = {"audit_revision": 3, "archive_sha256": digest(archive), "tracked_split_sha256": digest(manifest), "prepared_splits": expected}
        metadata.write_text(json.dumps(value), encoding="utf-8")
        args = argparse.Namespace(root=directory / "data", metadata=metadata, split_manifest=manifest, archive=archive,
                                 original_name="original", prepared_name="curated-v3", apply=False, allow_research_v2=False)
        return args, rows, value

    def test_dry_run_does_not_create_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _, _ = self.fixture(Path(folder))
            report = preparation.prepare(args)
            self.assertEqual(report["state"], "dry_run_verified")
            self.assertFalse(args.root.exists())
            self.assertFalse(report["trainingStarted"])

    def test_apply_and_reuse_verify_all_source_and_prepared_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _, _ = self.fixture(Path(folder))
            args.apply = True
            report = preparation.prepare(args)
            self.assertEqual(report["state"], "prepared_verified")
            self.assertEqual(preparation.prepare(args)["preparedHashes"], "verified")
            image = next((args.root / "curated-v3" / "train").glob("*.jpg"))
            image.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                preparation.prepare(args)

    def test_group_leak_and_legacy_selection_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args, rows, value = self.fixture(Path(folder))
            rows[1]["group_id"] = rows[0]["group_id"]
            with self.assertRaisesRegex(ValueError, "crossing"):
                preparation.check_selection(rows, value["prepared_splits"])
            value["audit_revision"] = 2
            args.metadata.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reviewed v3"):
                preparation.prepare(args)

    def test_oracle_roi_preflight_verifies_selected_archive_members_without_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            args, rows, value = self.fixture(Path(folder))
            for row in rows:
                row.update(cropPixelsXYWH=[0, 0, 4, 3], oracle_derived=True, roi_sha256="0" * 64)
            args.split_manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            value["tracked_split_sha256"] = digest(args.split_manifest)
            args.metadata.write_text(json.dumps(value), encoding="utf-8")
            report = preparation.prepare(args)
            self.assertEqual(report["evaluationScope"], "oracle_roi_auxiliary_only")
            self.assertEqual(report["selectedSourceImageHashes"], "verified_from_archive")
            self.assertEqual(report["preparedHashes"], "pending_materialization")
            self.assertFalse(args.root.exists())

    def test_offline_plan_cannot_be_started_by_job_runner(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            plan = directory / "plan.json"
            target = directory / "must-not-exist"
            plan.write_text(json.dumps({"kind": "offline_experiment_plan", "launchable": False,
                                       "run_dir": str(target), "argv": [sys.executable, "-c", "raise RuntimeError('must not run')"]}))
            result = subprocess.run([sys.executable, str(ROOT / "infra/windows/job_runner.py"), "--spec", str(plan)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not launchable", result.stderr)
            self.assertFalse(target.exists())

    def test_windows_and_posix_archive_escapes_are_rejected(self):
        for name in ("../escape", "/absolute", "C:/drive", "C:\\drive", "dir/../escape", "NUL.txt", "dir/file:stream", "name. "):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_name(name)

    def test_archive_symlinks_and_case_aliases_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            link = Path(folder) / "link.zip"
            with zipfile.ZipFile(link, "w") as zipped:
                info = zipfile.ZipInfo("link")
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                zipped.writestr(info, "../../outside")
            with self.assertRaisesRegex(ValueError, "symlink"):
                inspect_zip(link)
            aliases = Path(folder) / "aliases.zip"
            with zipfile.ZipFile(aliases, "w") as zipped:
                zipped.writestr("A/file.txt", "one")
                zipped.writestr("a/FILE.txt", "two")
            with self.assertRaisesRegex(ValueError, "aliased"):
                inspect_zip(aliases)

    def test_fresh_bundle_excludes_historical_launch_specs_and_oracle_plans(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "ml").mkdir()
            (root / "ml/current.py").write_text("pass")
            specs = root / "infra/windows/specs/curated-v3"
            specs.mkdir(parents=True)
            (specs / "old.plan.json").write_text('{"trainingSourceSha256":"obsolete"}')
            (specs / "old-run.json").write_text('{"argv":["python","old-training.py"]}')
            report = build(root, root / "fresh.zip", None, None)
            self.assertEqual(set(verify(root / "fresh.zip", report["sha256"])["files"]), {"ml/current.py"})

    def test_bundle_integrity_and_fresh_directory_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "ml").mkdir()
            (root / "ml" / "hello.py").write_text("print('fixture')\n")
            report = build(root, root / "code.zip", None, None)
            manifest = verify(root / "code.zip", report["sha256"])
            self.assertEqual(set(manifest["files"]), {"ml/hello.py"})
            extract_new(root / "code.zip", root / "unpacked", expected_sha256=report["sha256"])
            with self.assertRaises(FileExistsError):
                extract_new(root / "code.zip", root / "unpacked", expected_sha256=report["sha256"])
            tampered = root / "tampered.zip"
            with zipfile.ZipFile(root / "code.zip") as source, zipfile.ZipFile(tampered, "w") as target:
                target.writestr(MANIFEST, source.read(MANIFEST))
                target.writestr("ml/hello.py", "print('changed')\n")
            with self.assertRaisesRegex(ValueError, "file hash"):
                verify(tampered, digest(tampered))


if __name__ == "__main__":
    unittest.main()
