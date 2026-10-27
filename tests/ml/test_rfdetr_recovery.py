import tempfile
from pathlib import Path
import unittest
from ml.eval.recover_rfdetr import checkpoint_candidates


class RFCandidateTests(unittest.TestCase):
    def test_interrupted_fit_can_select_ema_without_missing_best_total(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'checkpoint_best_ema.pth').write_bytes(b'ema')
            (root/'last.ckpt').write_bytes(b'last')
            paths=checkpoint_candidates(root)
            self.assertEqual([p.name for p,_ in paths],['checkpoint_best_ema.pth','last.ckpt'])

    def test_identical_candidates_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('checkpoint_best_total.pth','checkpoint_best_ema.pth','last.ckpt'):(root/name).write_bytes(b'same')
            self.assertEqual(len(checkpoint_candidates(root)),1)


if __name__=='__main__':unittest.main()
