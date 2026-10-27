"""Queue behavior tests use real tiny CPU child processes, never SSH or GPUs."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from infra.windows.experiment_queue import ExperimentQueue, QueueLock, config_contract, parse_gpu_processes, read_json, signature, write_json
from infra.windows.watch_queue import snapshot, render


def fixture(root, *jobs):
    return {"schemaVersion": 1, "queue_id": "queue-test", "queue_dir": str(root / "queue"),
            "deadline_hours": .1, "reserve_minutes": 0, "poll_seconds": .05,
            "gpu_policy": "disabled", "experiments": list(jobs)}


def job(root, name, program="pass", **kwargs):
    value = {"id": name, "run_dir": str(root / name), "cwd": str(root),
             "estimated_minutes": .001, "timeout_minutes": .1,
             "steps": [{"name": "train", "argv": [sys.executable, "-c", program]}]}
    value.update(kwargs)
    return value


def execute(config):
    with redirect_stdout(io.StringIO()):
        return ExperimentQueue(config).run()


class QueueTests(unittest.TestCase):
    def test_failure_continues_next_experiment_and_restart_does_not_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "count.txt"
            append = f"from pathlib import Path; p=Path({str(marker)!r}); p.write_text((p.read_text() if p.exists() else '')+'X')"
            config = fixture(root, job(root, "broken", "import sys; print('owned error',file=sys.stderr);sys.exit(7)"),
                             job(root, "good", append))
            self.assertEqual(execute(config)["state"], "completed")
            self.assertEqual(read_json(root / "queue/jobs/broken.json")["state"], "failed")
            self.assertEqual(read_json(root / "queue/jobs/good.json")["state"], "succeeded")
            self.assertIn("owned error", (root / "broken/stderr.log").read_text())
            execute(config)
            self.assertEqual(marker.read_text(), "X")
            summary = snapshot(root / "queue")
            self.assertEqual(summary["counts"], {"failed": 1, "succeeded": 1})
            self.assertIn("failed=1", render(summary))

    def test_train_and_validation_steps_are_serial_and_reports_are_grouped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = []
            for index, valid_hash in enumerate(("valid-a", "valid-a", "valid-b")):
                name = "model-" + str(index)
                marker = root / (name + "-trained")
                report = root / (name + "-report.json")
                data = {"split": "validation_only", "validationAnnotationSha256": valid_hash,
                    "bestThreshold": .25, "selectionGroup": "ten_or_more_tiles",
                    "dataContract": {"splits": {"valid": {"imageTreeSha256": "pixels-" + valid_hash}}},
                    "thresholds": {"0.25": {"ten_or_more_tiles": {"hand_exact_match": index / 10,
                                     "model_exact": index / 10, "accepted_coverage": 1}}}}
                value = job(root, name)
                value["steps"] = [
                    {"name": "train", "argv": [sys.executable, "-c", f"from pathlib import Path;Path({str(marker)!r}).write_text('done')"]},
                    {"name": "validate", "argv": [sys.executable, "-c", f"from pathlib import Path;assert Path({str(marker)!r}).exists();Path({str(report)!r}).write_text({json.dumps(data)!r})"]}]
                value["report_paths"] = [str(report)]
                values.append(value)
            execute(fixture(root, *values))
            summary = read_json(root / "queue/summary.json")
            self.assertEqual(set(summary["validationCohorts"]), {"valid-a:pixels-valid-a", "valid-b:pixels-valid-b"})
            self.assertEqual([r["experiment"] for r in summary["validationCohorts"]["valid-a:pixels-valid-a"]], ["model-1", "model-0"])

    def test_timeout_controls_only_its_own_child_and_next_job_runs(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(10)"])
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = fixture(root, job(root, "timeout", "import time;time.sleep(10)", timeout_minutes=.002), job(root, "after"))
                execute(config)
                self.assertEqual(read_json(root / "queue/jobs/timeout.json")["state"], "timed_out")
                self.assertEqual(read_json(root / "queue/jobs/after.json")["state"], "succeeded")
                self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_estimated_budget_refuses_new_work_without_creating_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = job(root, "expensive", estimated_minutes=5)
            config = fixture(root, value)
            config["deadline_hours"] = .001
            execute(config)
            self.assertEqual(read_json(root / "queue/jobs/expensive.json")["state"], "not_started_budget")
            self.assertFalse((root / "expensive").exists())

    def test_restart_preserves_interrupted_run_and_does_not_blindly_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = job(root, "partial", "raise RuntimeError('must not execute')")
            config = fixture(root, value)
            run = root / "partial"
            run.mkdir()
            (run / "stdout.log").write_text("preserve this")
            receipt = {"id": "partial", "state": "running", "experiment_sha256": signature(value),
                       "run_dir": str(run), "child_pid": 999999999}
            write_json(root / "queue/jobs/partial.json", receipt)
            execute(config)
            self.assertEqual(read_json(root / "queue/jobs/partial.json")["state"], "interrupted")
            self.assertEqual((run / "stdout.log").read_text(), "preserve this")

    def test_live_previous_pid_causes_bounded_wait_without_duplicate_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = job(root, "partial")
            config = fixture(root, value)
            config["wait_timeout_minutes"] = .002
            write_json(root / "queue/jobs/partial.json", {"id": "partial", "state": "running",
                "experiment_sha256": signature(value), "run_dir": value["run_dir"], "child_pid": os.getpid()})
            with self.assertRaisesRegex(RuntimeError, "Waiting timed out"):
                execute(config)
            self.assertEqual(read_json(root / "queue/state.json")["state"], "blocked")
            self.assertFalse((root / "partial").exists())

    def test_changed_config_cannot_reuse_queue_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixture(root, job(root, "a"))
            execute(config)
            config["experiments"][0]["steps"][0]["argv"][-1] = "raise RuntimeError('never run')"
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                execute(config)

    def test_os_lock_blocks_a_second_coordinator(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queue.lock"
            with QueueLock(path), self.assertRaisesRegex(RuntimeError, "live coordinator"):
                with QueueLock(path):
                    self.fail("lock was not exclusive")
            with QueueLock(path):
                pass

    def test_stale_external_status_and_busy_compute_are_checked_before_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old-status.json"
            write_json(old, {"state": "running", "child_pid": 999999999})
            config = fixture(root, job(root, "after"))
            config.update(wait_for_status=[str(old)], gpu_policy="exclusive")
            samples = [{"blocking_processes": [{"pid": 42, "name": "other-gpu"}], "allowed_background_processes": []},
                       {"blocking_processes": [], "allowed_background_processes": []}]
            def sample(*_):
                return samples.pop(0) if len(samples) > 1 else samples[0]
            with patch("infra.windows.experiment_queue.query_gpu_processes", side_effect=sample):
                execute(config)
            self.assertEqual(read_json(root / "queue/jobs/after.json")["state"], "succeeded")
            events = [json.loads(line) for line in (root / "queue/events.jsonl").read_text().splitlines()]
            self.assertTrue(any(e.get("detail", {}).get("state") == "orphaned_status_process_exited" for e in events))
            self.assertEqual(len([e for e in events if e["event"] == "wait_released"]), 2)

    def test_wddm_graphics_and_exact_desktop_paths_do_not_block_cuda_queue(self):
        xml = r"""<nvidia_smi_log><gpu id="GPU0"><processes>
        <process_info><pid>11</pid><type>G</type><process_name>C:\Windows\System32\dwm.exe</process_name></process_info>
        <process_info><pid>12</pid><type>C+G</type><process_name>C:\WINDOWS\Explorer.EXE</process_name></process_info>
        <process_info><pid>13</pid><type>C+G</type><process_name>C:\Python311\python.exe</process_name></process_info>
        <process_info><pid>14</pid><type>C</type><process_name>C:\Unknown\worker.exe</process_name></process_info>
        <process_info><pid>15</pid><process_name>C:\Unknown\other.exe</process_name></process_info>
        </processes></gpu></nvidia_smi_log>"""
        result = parse_gpu_processes(xml, ["c:/windows/explorer.exe"])
        self.assertEqual([p["pid"] for p in result["graphics_processes"]], [11])
        self.assertEqual([p["pid"] for p in result["allowed_background_processes"]], [12])
        self.assertEqual([p["pid"] for p in result["blocking_processes"]], [13, 14, 15])
        self.assertEqual([p["pid"] for p in parse_gpu_processes(xml)["blocking_processes"]], [12, 13, 14, 15])

    def test_gpu_allowlist_rejects_generic_names_ml_runtimes_and_unknown_process_data(self):
        for path in ("explorer.exe", "C:/Python311/python.exe", "C:/Tools/torchrun.exe"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                parse_gpu_processes('<nvidia_smi_log><gpu><processes/></gpu></nvidia_smi_log>', [path])
        for xml in ('<nvidia_smi_log/>', '<nvidia_smi_log><gpu/></nvidia_smi_log>',
                    '<nvidia_smi_log><gpu><processes>N/A</processes></gpu></nvidia_smi_log>'):
            with self.subTest(xml=xml), self.assertRaises(ValueError):
                parse_gpu_processes(xml)
        self.assertEqual(parse_gpu_processes('<nvidia_smi_log><gpu><processes/></gpu></nvidia_smi_log>')["blocking_processes"], [])

    def test_failed_dependency_skips_only_dependent_job_and_keeps_independent_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixture(root, job(root, "detector", "raise SystemExit(3)"),
                             job(root, "classifier"),
                             job(root, "cascade", depends_on=["detector", "classifier"]),
                             job(root, "independent"))
            execute(config)
            self.assertEqual(read_json(root / "queue/jobs/cascade.json")["state"], "skipped_dependency")
            self.assertFalse((root / "cascade").exists())
            self.assertEqual(read_json(root / "queue/jobs/independent.json")["state"], "succeeded")
            self.assertEqual(read_json(root / "queue/state.json")["result"], "completed_with_failures")

    def test_succeeded_dependencies_allow_end_to_end_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixture(root, job(root, "detector"), job(root, "classifier"),
                             job(root, "cascade", depends_on=["detector", "classifier"]))
            execute(config)
            self.assertEqual(read_json(root / "queue/jobs/cascade.json")["state"], "succeeded")

    def test_forward_duplicate_and_unknown_dependencies_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dependencies in (["later"], ["missing"], ["first", "first"], "first"):
                with self.subTest(dependencies=dependencies), self.assertRaisesRegex(ValueError, "Dependencies"):
                    config_contract(fixture(root, job(root, "first"), job(root, "second", depends_on=dependencies), job(root, "later")))

    def test_budget_hard_limit_and_duplicate_experiment_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixture(root, job(root, "same"), job(root, "same"))
            with self.assertRaisesRegex(ValueError, "unique"):
                config_contract(config)
            config = fixture(root, job(root, "same"))
            config["deadline_hours"] = 25
            with self.assertRaisesRegex(ValueError, "24"):
                config_contract(config)


if __name__ == "__main__":
    unittest.main()
