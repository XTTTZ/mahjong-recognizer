"""Recovery accepts signed terminal checkpoints without changing run completion history."""
import json
from pathlib import Path
import tempfile
import unittest
from ml.eval.recover_terminal import checked_candidates, checked_run
from ml.train.yolo_train import checkpoint_receipt


class TerminalRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        contract={'classes':{'0':'tile'},'splits':{'train':{'annotationSha256':'train'},'valid':{'annotationSha256':'valid'}}}
        run={'classes':contract['classes'],'resumeSignature':{'data':contract['splits']},
             'annotationSha256':'train','validationAnnotationSha256':'valid','finishedAt':None}
        path=root/'run.json';path.write_text(json.dumps(run))
        status=root/'status.json';status.write_text(json.dumps({'state':'timed_out','run_dir':str(root),'child_pid':None,'finished_at':'2026-10-27T06:00:00Z'}))
        for name in ('best.pt','last.pt'):
            (root/name).write_bytes(b'complete checkpoint')
            checkpoint_receipt(root/name,run['resumeSignature'],run['classes'])
        return contract,path,status

    def test_explicit_terminal_proof_preserves_original_run_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,path,status=self.fixture(root);before=path.read_bytes()
            run,proof,candidates=checked_candidates(path,root/'best.pt',contract,status)
            self.assertIsNone(run['finishedAt'])
            self.assertEqual(proof['trainingCompletion'],'timed_out_checkpoint_recovery')
            self.assertEqual(len(candidates),1)
            self.assertEqual(path.read_bytes(),before)

    def test_unfinished_without_terminal_proof_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            contract,path,status=self.fixture(Path(d))
            with self.assertRaises(ValueError):checked_run(path,contract)

    def test_live_or_wrong_run_receipt_is_rejected(self):
        for change in ({'state':'running'},{'child_pid':123},{'run_dir':'elsewhere'},{'finished_at':None}):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                contract,path,status=self.fixture(Path(d))
                value=json.loads(status.read_text());value.update(change);status.write_text(json.dumps(value))
                with self.assertRaises(ValueError):checked_run(path,contract,status)

    def test_torn_checkpoint_cannot_be_recovered(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,path,status=self.fixture(root)
            (root/'best.pt').write_bytes(b'torn save')
            with self.assertRaises(ValueError):checked_candidates(path,root/'best.pt',contract,status)


if __name__=='__main__':unittest.main()
