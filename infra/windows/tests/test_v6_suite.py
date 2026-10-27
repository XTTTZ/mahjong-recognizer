"""Cold v6 preparation tests; no SSH, model execution, training, or background launch."""
import contextlib,io,json,os,tempfile,unittest,warnings
from pathlib import Path
from infra.windows.prepare_refinement_suite import make_suite,write_plan,cold_bundle
from infra.windows.offline_bundle import verify
from infra.windows.prepare_v6_host import prepare
from infra.windows.validate_cli import validate

ROOT=Path(__file__).resolve().parents[3]

def fixture_configuration():
    # Synthetic pins/assets, independent of any user's machine snapshots.
    return {'expected_environment':{'python':'3.11.17','packages':{'fixture-runtime':'1.0'}},
            'parent_assets':[{'path':'D:/fixture/parent.pt','sha256':'a'*64}]}


class RefinementSuiteTests(unittest.TestCase):
    def test_eight_independent_bounded_branches_share_parents_and_reserve_validation_time(self):
        suite=make_suite(ROOT,**fixture_configuration());config=suite['queueConfig'];jobs=config['experiments']
        self.assertEqual(config['deadline_hours'],12);self.assertEqual(config['reserve_minutes'],60)
        self.assertEqual(len(jobs),8);self.assertEqual(sum(len(j['steps'])for j in jobs),16)
        self.assertFalse(suite['trainingStarted']);self.assertEqual(suite['preparationStatus'],'offline_frozen_remote_v6_unverified')
        self.assertLess(sum(j['estimated_minutes']for j in jobs)+60,12*60)
        for job in jobs:
            argv=job['steps'][0]['argv'];self.assertEqual(argv[argv.index('--epochs')+1],'12')
            self.assertIn('--max-train-minutes',argv)
            self.assertEqual(job['steps'][1]['env']['CUDA_VISIBLE_DEVICES'],'-1')
            self.assertNotIn('depends_on',job)
            self.assertGreater(job['timeout_minutes'],job['steps'][0]['timeout_minutes'])
        parents=[j['steps'][0]['argv'][j['steps'][0]['argv'].index('--parent-checkpoint')+1]for j in jobs]
        self.assertEqual(len(set(parents[:3])),1);self.assertEqual(len(set(parents[3:5])),1)
        self.assertEqual(len(set(parents[5:])),1)
        for job in jobs[5:]:self.assertIn('tile-detector-v26-960/run.json', ' '.join(job['steps'][1]['argv']))

    def test_all_sixteen_commands_parse_then_stop_before_entrypoint_work(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            for job in make_suite(ROOT,**fixture_configuration())['queueConfig']['experiments']:
                for step in job['steps']:
                    with self.subTest(job=job['id'],step=step['name']):
                        self.assertEqual(validate(step['argv'])['state'],'argv_parsed_only')

    def test_only_submit_script_contains_explicit_start_and_bundle_verifies(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp);plan=path/'v6-12h'
            write_plan(ROOT,plan,**fixture_configuration())
            self.assertIn('--start',(plan/'submit.cmd').read_text())
            for name in ('prepare.cmd','check.cmd','watch.cmd'):
                self.assertNotIn('--start',(plan/name).read_text())
            self.assertIn('setlocal',(plan/'check.cmd').read_text())
            self.assertIn('set CUDA_VISIBLE_DEVICES=-1',(plan/'prepare.cmd').read_text())
            receipt=cold_bundle(ROOT,plan,path/'cold.zip');manifest=verify(path/'cold.zip',receipt['sha256'])
            self.assertIn('prepared-suites/v6-12h/suite.json',manifest['files'])
            self.assertNotIn('infra/windows/verified/environment.json',manifest['files'])
            self.assertTrue(all(not n.startswith('infra/windows/verified/')or n.startswith('infra/windows/verified/v6/')for n in manifest['files']))
            self.assertFalse(receipt['remoteV6CheckPassed']);self.assertFalse(receipt['trainingStarted'])

    @unittest.skipIf(os.name=='nt','Non-Windows guard')
    def test_host_preparation_cannot_run_on_the_mac(self):
        with self.assertRaisesRegex(RuntimeError,'Windows'):prepare({})


if __name__=='__main__':unittest.main()
