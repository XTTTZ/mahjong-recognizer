"""Regression tests for progress rendering as epoch history grows."""
from copy import deepcopy
import csv
from pathlib import Path
import tempfile
import unittest
from infra.windows.watch_queue import render, epoch_metrics


class WatchQueueTests(unittest.TestCase):
    def snapshot(self, rows):
        return {'queue': {'queue_id': 'test-12h', 'state': 'running',
                         'deadline_epoch_seconds': 1793105015, 'current_experiment': 'yolo',
                         'current_step': 'train', 'child_pid': 6528},
                'counts': {'running': 1, 'pending': 16}, 'total': 17,
                'plannedRemainingMinutes': 2016, 'currentMaxEpochs': 50,
                'currentEpochs': rows}

    def test_multiple_epochs_and_both_metric_formats_preserve_snapshot(self):
        value = self.snapshot([
            {'epoch': '0', 'train/loss': '5.46953', 'mAP50': '.53312', 'mAP50_95': '.43102'},
            {'epoch': '1', 'train/loss': '2.72897', 'mAP50': '.73167', 'mAP50_95': '.60842'},
            {'epoch': '2', 'train/loss': '2.25881', 'val/mAP_50': '.75692', 'val/mAP_50_95': '.62730'}])
        before = deepcopy(value)
        output = render(value)
        self.assertIn('epoch=1/50  loss=5.46953  mAP50=0.53312  mAP50_95=0.43102', output)
        self.assertIn('epoch=2/50  loss=2.72897  mAP50=0.73167  mAP50_95=0.60842', output)
        self.assertIn('epoch=3/50  loss=2.25881  mAP50=0.75692  mAP50_95=0.62730', output)
        self.assertEqual(value, before)

    def test_zero_metrics_remain_visible(self):
        value = self.snapshot([{'epoch': '0', 'train/loss': 0, 'val/mAP_50': 0, 'val/mAP_50_95': '0'}])
        output = render(value)
        self.assertIn('loss=0.00000', output)
        self.assertIn('mAP50=0.00000', output)
        self.assertIn('mAP50_95=0.00000', output)

    def test_complementary_rf_rows_render_multiple_complete_epochs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with (path/'metrics.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['epoch','train/loss','val/mAP_50'])
                writer.writeheader()
                writer.writerows([{'epoch': 0, 'train/loss': 8.7}, {'epoch': 0, 'val/mAP_50': .2},
                                  {'epoch': 1, 'train/loss': 7.9}, {'epoch': 1, 'val/mAP_50': .3}])
            output = render(self.snapshot(epoch_metrics(path)))
            self.assertIn('epoch=1/50  loss=8.70000  mAP50=0.20000', output)
            self.assertIn('epoch=2/50  loss=7.90000  mAP50=0.30000', output)


if __name__ == '__main__': unittest.main()
