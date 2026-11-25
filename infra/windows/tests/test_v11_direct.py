import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from infra.windows.experiment_queue import ExperimentQueue, read_json
from infra.windows.launch_prepared_queue import bind_new_window
from infra.windows.prepare_direct_suite_v11 import RECORD, ROOT, make_suite
from infra.windows.validate_cli import validate
from ml.eval.direct_latency import checked_control


class DirectSuiteTest(unittest.TestCase):
    def test_new_face_head_all_jobs_complete_held_out_validation_and_paired_benchmark(self):
        suite = make_suite()
        public_record = suite['codeRoot'] + '/' + RECORD
        sources = [asset['path'] for asset in suite['sourceFiles']]
        self.assertIn(public_record, sources)
        self.assertFalse(any('/docs/models/runs/' in source for source in sources))
        manifest = json.loads((ROOT / RECORD).read_text(encoding='utf-8'))
        self.assertEqual(set(manifest), {'detector', 'classifier'})
        for component in manifest.values():
            self.assertEqual(set(component), {'checkpoint', 'run', 'receipt'})
            for asset in component.values():
                self.assertEqual(set(asset), {'path', 'sha256'})
                self.assertRegex(asset['sha256'], r'^[0-9a-f]{64}$')
                self.assertIn(asset, suite['requiredFiles'])
        self.assertTrue(suite['requiresNewWindow'])
        self.assertFalse(suite['trainingStarted'])
        config = suite['queueConfig']
        self.assertEqual(config['deadline_hours'], 12)
        self.assertLessEqual(sum(j['timeout_minutes'] for j in config['experiments']) + config['reserve_minutes'], 720)
        for job in config['experiments']:
            train, selection, benchmark = [s['argv'] for s in job['steps']]
            self.assertIn('--face-manifest', train)
            self.assertNotIn('--background-review', train)
            self.assertIn('--locked-operating-point', selection)
            self.assertIn('ml.eval.direct_latency', benchmark)
            self.assertEqual(benchmark[benchmark.index('--control-record') + 1], public_record)
            for argv in (train, selection, benchmark):
                self.assertFalse(validate(argv)['trainingStarted'])

    def test_benchmark_checks_the_frozen_checkpoint_run_and_receipt(self):
        data = {'train': {'annotationSha256': 'train'}, 'valid': {'annotationSha256': 'valid'}}
        classes = {'0': 'tile'}
        signature = {'data': data}
        run = {'resumeSignature': signature, 'classes': classes, 'finishedAt': '2026-11-09',
               'annotationSha256': 'train', 'validationAnnotationSha256': 'valid'}
        contract = {'splits': data, 'classes': classes}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory, 'best.pt')
            checkpoint.write_bytes(b'fixture')
            receipt = {'checkpointSha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       'signatureSha256': hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
                       'classes': classes}
            run_path, receipt_path = Path(directory, 'run.json'), Path(str(checkpoint) + '.receipt.json')
            run_path.write_text(json.dumps(run), encoding='utf-8')
            receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
            component = {key: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                         for key, path in [('checkpoint', checkpoint), ('run', run_path), ('receipt', receipt_path)]}
            self.assertEqual(checked_control(component, contract), (checkpoint, run))
            for key in component:
                with self.subTest(key=key):
                    tampered = {**component, key: {**component[key], 'sha256': '0' * 64}}
                    with self.assertRaisesRegex(ValueError, key + ' hash changed'):
                        checked_control(tampered, contract)

    def test_no_window_no_replay_or_late_restart(self):
        now = datetime(2026, 11, 25, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            suite = make_suite()
            q = suite['queueConfig']
            q['queue_dir'] = directory+'/unused'
            for until in ('', '2026-11-25T02:00:00', '2026-11-25T00:00:00Z', '2026-11-25T14:00:00Z'):
                with self.assertRaises(ValueError):
                    bind_new_window(suite, until, now.timestamp())
            with self.assertRaises(ValueError):
                ExperimentQueue(q, clock=now.timestamp).run()
            self.assertFalse(Path(q['queue_dir']).exists())
            bind_new_window(suite, '2026-11-25T02:00:00Z', now.timestamp())
            self.assertEqual(q['authorized_until_epoch_seconds'], now.timestamp()+3600)
            with self.assertRaises(ValueError):
                ExperimentQueue(q, clock=lambda: now.timestamp()+3601).run()
            Path(q['queue_dir']).mkdir()
            with self.assertRaises(FileExistsError):
                bind_new_window(suite, '2026-11-25T02:00:00Z', now.timestamp())


if __name__ == '__main__':
    unittest.main()
