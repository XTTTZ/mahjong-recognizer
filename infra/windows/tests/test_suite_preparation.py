"""Preparation/launch guard tests never submit WMI or train models."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

from infra.windows.launch_prepared_queue import last_json, submit, validate_suite
from infra.windows.prepare_experiment_suite import make_suite

ROOT = Path(__file__).resolve().parents[3]


def fixture(root):
    code = root / 'source.py'
    asset = root / 'weights.bin'
    code.write_text('pass\n')
    asset.write_bytes(b'verified-pretrained-asset')
    return {'schemaVersion': 1, 'kind': 'prepared_experiment_suite', 'trainingStarted': False,
            'assetsVerifiedByPreparation': True, 'codeRoot': str(root),
            'sourceFiles': [{'path': str(code), 'sha256': hashlib.sha256(code.read_bytes()).hexdigest()}],
            'pythonPaths': [sys.executable], 'environmentSnapshots': [{'python': sys.executable, 'version': platform.python_version(), 'packages': {}}],
            'requiredFiles': [{'path': str(asset), 'sha256': hashlib.sha256(asset.read_bytes()).hexdigest()}],
            'cpuPreflightCommands': [],
            'queueConfig': {'schemaVersion': 1, 'queue_id': 'check-only', 'queue_dir': str(root / 'queue'),
                            'deadline_hours': 5, 'gpu_policy': 'disabled',
                            'experiments': [{'id': 'a', 'cwd': str(root), 'run_dir': str(root / 'run'),
                                             'estimated_minutes': 1, 'timeout_minutes': 2,
                                             'steps': [{'name': 'train', 'argv': [sys.executable, '-c', 'raise RuntimeError("must never execute")']}]}]}}


def fake_runtime(argv, **kwargs):
    if any(str(a).endswith('validate_cli.py') for a in argv):
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps({'state':'argv_parsed_only','trainingStarted':False,'module':'ml.train.stub'}), stderr='')
    output = json.dumps({'python': platform.python_version(), 'packages': {}}) if '-c' in argv else 'No broken requirements found.\n'
    return subprocess.CompletedProcess(argv, 0, stdout=output, stderr='')


class SuiteTests(unittest.TestCase):
    def test_catalog_has_independent_architectures_and_ordered_cascade_dependencies(self):
        for hours in (5, 12, 24):
            suite = make_suite(ROOT, hours=hours, queue_id='v5-' + str(hours) + 'h')
            jobs = suite['queueConfig']['experiments']
            self.assertEqual(len(jobs), 17)
            self.assertEqual(suite['queueConfig']['deadline_hours'], hours)
            self.assertFalse(suite['trainingStarted'])
            self.assertEqual(jobs[3]['depends_on'], [jobs[1]['id'], jobs[2]['id']])
            self.assertIn('ml.eval.evaluate_yolo_candidates', jobs[3]['steps'][0]['argv'])
            for check in suite['cpuPreflightCommands']:
                self.assertIn('--preflight-only', check['argv'])
            self.assertEqual(sum('--verify-images' in c['argv'] for c in suite['cpuPreflightCommands']), 2)

    def test_yolo_direct_and_cascade_selection_use_full_image_checkpoint_candidates(self):
        suite = make_suite(ROOT)
        steps = [step for job in suite['queueConfig']['experiments'] for step in job['steps']]
        candidate_steps = [step for step in steps if 'ml.eval.evaluate_yolo_candidates' in step['argv']]
        self.assertEqual(len(candidate_steps), 5)
        paired = [step for step in candidate_steps if '--classifier-run' in step['argv']]
        self.assertEqual(len(paired), 2)
        for step in candidate_steps:
            self.assertIn('--run', step['argv'])
            self.assertIn('--checkpoint', step['argv'])
            self.assertNotIn('--detector-run', step['argv'])
            self.assertNotIn('--detector', step['argv'])
        # No orphan raw evaluator can bypass the candidate selection contract.
        self.assertFalse(any('ml.eval.yolo_evaluate' in step['argv'] or 'ml.eval.cascade_evaluate' in step['argv'] for step in steps))

    def test_every_catalog_step_parses_and_stops_before_loading_data_or_models(self):
        from infra.windows.validate_cli import validate
        suite = make_suite(ROOT)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for job in suite['queueConfig']['experiments']:
                for step in job['steps']:
                    with self.subTest(experiment=job['id'], step=step['name']):
                        receipt = validate(step['argv'])
                        self.assertEqual(receipt['state'], 'argv_parsed_only')
                        self.assertFalse(receipt['trainingStarted'])

    def test_check_only_reads_evidence_without_creating_queue_run_or_launch_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = fixture(root)
            before = set(root.iterdir())
            with patch('infra.windows.launch_prepared_queue.subprocess.run', side_effect=fake_runtime):
                report = validate_suite(suite)
            self.assertEqual(set(root.iterdir()), before)
            self.assertEqual(report['state'], 'prepared_checks_passed')
            self.assertFalse(report['trainingStarted'])
            self.assertFalse((root / 'queue').exists())
            self.assertFalse((root / 'run').exists())

    def test_missing_pins_and_source_or_asset_mutations_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = fixture(root)
            suite['assetsVerifiedByPreparation'] = False
            with self.assertRaisesRegex(ValueError, 'asset-manifest'):
                validate_suite(suite)
            suite['assetsVerifiedByPreparation'] = True
            (root / 'source.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Frozen source'):
                validate_suite(suite)
            suite['sourceFiles'] = []
            with patch('infra.windows.launch_prepared_queue.subprocess.run', side_effect=fake_runtime):
                (root / 'weights.bin').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    validate_suite(suite)

    def test_runtime_pins_and_derivative_readiness_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = fixture(root)
            suite['environmentSnapshots'][0]['packages'] = {'torch': 'wrong'}
            with patch('infra.windows.launch_prepared_queue.subprocess.run', side_effect=fake_runtime), self.assertRaisesRegex(ValueError, 'drifted'):
                validate_suite(suite)
            suite['environmentSnapshots'][0]['packages'] = {}
            suite['cpuPreflightCommands'] = [{'name': 'crops', 'requireDerivativeReady': True,
                'argv': [sys.executable, '-m', 'ml.data.prepare_crops', '--preflight-only']}]
            def runner(argv, **kwargs):
                if 'ml.data.prepare_crops' in argv:
                    return subprocess.CompletedProcess(argv, 0, stdout='{"derivativeReady": false}', stderr='')
                return fake_runtime(argv, **kwargs)
            with patch('infra.windows.launch_prepared_queue.subprocess.run', side_effect=runner), self.assertRaisesRegex(ValueError, 'not ready'):
                validate_suite(suite)

    def test_training_argv_cannot_be_run_as_a_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = fixture(Path(temporary))
            suite['cpuPreflightCommands'] = [{'name': 'bad', 'argv': [sys.executable, '-m', 'ml.train.yolo_train']}]
            with patch('infra.windows.launch_prepared_queue.subprocess.run', side_effect=fake_runtime), self.assertRaisesRegex(ValueError, 'explicit ML/data preflight'):
                validate_suite(suite)

    @unittest.skipIf(os.name == 'nt', 'non-Windows launch rejection')
    def test_submission_does_not_run_on_the_mac(self):
        with self.assertRaisesRegex(RuntimeError, 'Windows'):
            submit(Path('unused'), {}, {})

    def test_json_receipt_parser_requires_an_object(self):
        self.assertEqual(last_json('log\n{"ok":true}\n'), {'ok': True})
        self.assertEqual(last_json('log\n{\n  "ok": true,\n  "nested": {"x":1}\n}\n'), {'ok': True, 'nested': {'x': 1}})
        with self.assertRaises(ValueError):
            last_json('logging only\n[]')


if __name__ == '__main__':
    unittest.main()
