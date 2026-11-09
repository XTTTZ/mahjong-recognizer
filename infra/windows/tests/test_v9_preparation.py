from pathlib import Path
import json
import tempfile
import unittest
from infra.windows.prepare_portable_suite_v9 import make_suite, write_plan, RECIPES
from infra.windows.validate_cli import validate

ROOT=Path(__file__).resolve().parents[3]
class V9Preparation(unittest.TestCase):
    def test_budget_independence_fixed_quality_and_entrypoints(self):
        suite=make_suite(ROOT); q=suite['queueConfig']
        self.assertEqual(q['deadline_hours'],12);self.assertEqual(len(q['experiments']),7)
        self.assertLessEqual(sum(j['timeout_minutes'] for j in q['experiments'])+q['reserve_minutes'],720)
        self.assertFalse(suite['trainingStarted']);self.assertFalse(suite['submitted']);self.assertEqual(q['allowed_gpu_process_paths'],[])
        for job in q['experiments']:
            self.assertNotIn('depends_on',job)
            train,evaluate=job['steps']
            self.assertEqual(evaluate['env']['CUDA_VISIBLE_DEVICES'],'-1')
            self.assertIn('--locked-operating-point',evaluate['argv'])
            self.assertEqual(sum(s['timeout_minutes'] for s in job['steps']),job['timeout_minutes'])
            for step in job['steps']:self.assertFalse(validate(step['argv'])['trainingStarted'])
            name=job['id'].removeprefix(q['queue_id']+'-')
            check=next(c for c in suite['cpuPreflightCommands'] if c['name']==name)
            self.assertEqual(check['argv'],train['argv']+['--preflight-only'])
        self.assertEqual(suite['budget']['estimatedMinutes'],645)
        self.assertEqual(suite['budget']['sumJobHardCapsMinutes'],680)
    def test_only_submit_can_start_and_plan_is_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'prepared';write_plan(ROOT,out)
            self.assertIn('--start',(out/'submit.cmd').read_text())
            for name in ('prepare.cmd','check.cmd','watch.cmd'):self.assertNotIn('--start',(out/name).read_text())
            with self.assertRaises(FileExistsError):write_plan(ROOT,out)
    def test_nano_does_not_reuse_larger_architecture_parent(self):
        suite=make_suite(ROOT)
        for job in suite['queueConfig']['experiments']:
            argv=job['steps'][0]['argv']
            if '-v8n-' in job['id']:
                self.assertIn('ml.train.yolo_train_v9',argv);self.assertNotIn('--parent-checkpoint',argv)
                self.assertTrue(argv[argv.index('--weights')+1].endswith('/yolov8n.pt'))
if __name__=='__main__':unittest.main()
