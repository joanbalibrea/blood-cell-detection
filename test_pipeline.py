"""Focused regression checks for label validation, leakage detection and streamed metrics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import audit_dataset
import stream_logs

class DatasetAuditTests(unittest.TestCase):
    def test_invalid_box_and_cross_split_duplicate_are_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for i, split in enumerate(('train', 'val', 'test')):
                (root / 'images' / split).mkdir(parents=True)
                (root / 'labels' / split).mkdir(parents=True)
                # Train and validation deliberately contain identical images.
                Image.new('RGB', (20, 20), 'red' if i < 2 else 'blue').save(root / 'images' / split / 'cell.png')
                (root / 'labels' / split / 'cell.txt').write_text('0 0.5 0.5 0.2 0.2\n')
            (root / 'labels/test/cell.txt').write_text('3 0.5 0.5 0.2 0.2\n0 0.95 0.5 0.2 0.2\n')
            report = audit_dataset.audit(root)
            self.assertEqual(len(report['errors']), 2)
            self.assertEqual(len(report['cross_split_duplicates']), 1)
            self.assertEqual(report['total_objects'], 2)

    def test_missing_and_orphan_labels_are_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ('train', 'val', 'test'):
                (root / 'images' / split).mkdir(parents=True)
                (root / 'labels' / split).mkdir(parents=True)
            Image.new('RGB', (20, 20)).save(root / 'images/train/missing.png')
            (root / 'labels/train/orphan.txt').write_text('')
            self.assertEqual(len(audit_dataset.audit(root)['errors']), 2)

class StreamTests(unittest.TestCase):
    def test_stream_publishes_actual_metrics_and_deduplicates_epoch(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(stream_logs, 'ROOT', Path(temp)):
            root = Path(temp)
            (root / 'dashboard/data').mkdir(parents=True)
            history = {}
            state = {'phase': 'training', 'epoch': 1, 'metrics': {'metrics/mAP50(B)': 0.4,
                     'metrics/mAP50-95(B)': 0.2}, 'train_loss': {'train/box_loss': 2.0, 'train/cls_loss': 1.0}}
            stream_logs.publish(state, history)
            stream_logs.publish(state, history)
            self.assertEqual(len(history), 1)
            self.assertEqual(json.loads((root / 'dashboard/data/progress.json').read_text()), state)
            self.assertEqual(len((root / 'dashboard/data/results.csv').read_text().splitlines()), 2)
            # Completed history must supersede live-stream data.
            final = root / 'outputs/runs/blood_cells/results.csv'
            final.parent.mkdir(parents=True)
            final.write_text('complete')
            before = (root / 'dashboard/data/results.csv').read_text()
            stream_logs.publish(dict(state, epoch=2), history)
            self.assertEqual((root / 'dashboard/data/results.csv').read_text(), before)

if __name__ == '__main__':
    unittest.main()
