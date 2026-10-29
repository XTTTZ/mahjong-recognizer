"""Real tiny CPU children exercise queue deadline/continuation; never execute ML."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from infra.windows.experiment_queue import ExperimentQueue,read_json
from infra.windows.tests.test_experiment_queue import fixture,job


class V7BudgetTests(unittest.TestCase):
    def test_global_deadline_kills_owned_step_and_refuses_remaining_jobs(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            first=job(root,'first','import time;time.sleep(10)',estimated_minutes=.00001,timeout_minutes=10)
            after=job(root,'after','raise RuntimeError("must not start")',estimated_minutes=.00001)
            config=fixture(root,first,after);config['deadline_hours']=.0001
            with redirect_stdout(io.StringIO()):state=ExperimentQueue(config).run()
            self.assertEqual(state['state'],'completed')
            self.assertEqual(read_json(root/'queue/jobs/first.json')['state'],'timed_out')
            self.assertEqual(read_json(root/'queue/jobs/after.json')['state'],'not_started_budget')
            self.assertFalse((root/'after').exists())

    def test_wall_clock_rollback_cannot_extend_monotonic_deadline(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);clock=[1000.];mono=[20.]
            queue=ExperimentQueue(fixture(root,job(root,'one')),clock=lambda:clock[0],monotonic=lambda:mono[0])
            queue.state={'deadline_epoch_seconds':1100.};queue.deadline_monotonic=120.
            self.assertEqual(queue.remaining(),100.)
            clock[0]=0.;mono[0]=120.
            self.assertEqual(queue.remaining(),0.)


if __name__=='__main__':unittest.main()
