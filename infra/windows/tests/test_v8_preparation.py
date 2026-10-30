from pathlib import Path
import tempfile
import unittest
from infra.windows.prepare_portable_suite_v8 import make_suite,write_plan

ROOT=Path(__file__).resolve().parents[3]
class PortablePlanTests(unittest.TestCase):
    def test_budget_and_independent_jobs(self):
        s=make_suite(ROOT);q=s['queueConfig']
        self.assertFalse(s['trainingStarted']);self.assertFalse(s['newWindowAuthorized'])
        self.assertEqual(q['deadline_hours'],12);self.assertEqual(len(q['experiments']),7)
        self.assertLessEqual(sum(j['estimated_minutes'] for j in q['experiments'])+q['reserve_minutes'],720)
        self.assertEqual(q['allowed_gpu_process_paths'],[])
        for j in q['experiments']:
            argv=j['steps'][0]['argv'];self.assertEqual(argv[argv.index('--epochs')+1],'30')
            self.assertNotIn('int8',j['id']);self.assertEqual(j['steps'][1]['env']['CUDA_VISIBLE_DEVICES'],'-1')
            check=next(c for c in s['cpuPreflightCommands'] if c['name']==j['id'].removeprefix(q['queue_id']+'-'))
            self.assertEqual(check['argv'],argv+['--preflight-only'])
    def test_only_submit_is_live_and_reject_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'v8'
            write_plan(ROOT,out)
            self.assertIn('--start',(out/'submit.cmd').read_text())
            for name in ('check.cmd','prepare.cmd','watch.cmd'):self.assertNotIn('--start',(out/name).read_text())
            with self.assertRaises(FileExistsError):write_plan(ROOT,out)
if __name__=='__main__':unittest.main()
