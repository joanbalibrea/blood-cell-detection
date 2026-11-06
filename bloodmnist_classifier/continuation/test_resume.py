"""Focused restoration/cache checks before spending GPU time on continuation."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'training')); sys.path.insert(0, str(HERE))
from common import build_model
from resume import restore_training_state, state_fingerprint, CachedImageFolder


class ResumeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        previous = ROOT / 'outputs/download/v2/cellscope'
        cls.checkpoint = torch.load(previous / 'last.pt', map_location='cpu', weights_only=True)
        cls.history = json.loads((previous / 'history.json').read_text())

    def components(self):
        model = build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=3)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        return model, optimizer, scheduler, scaler

    def test_real_epoch33_restores_parameters_moments_and_missing_scheduler_step(self):
        model, optimizer, scheduler, scaler = self.components()
        next_epoch = restore_training_state(model, optimizer, scheduler, scaler, self.checkpoint, self.history)
        self.assertEqual(next_epoch, 34)
        self.assertEqual(scheduler.last_epoch, 33)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .00001875)
        key = 'features.0.0.weight'
        self.assertTrue(torch.equal(model.state_dict()[key], self.checkpoint['state_dict'][key]))
        parameter = next(model.parameters())
        original_id = self.checkpoint['optimizer']['param_groups'][0]['params'][0]
        self.assertTrue(torch.equal(optimizer.state[parameter]['exp_avg'], self.checkpoint['optimizer']['state'][original_id]['exp_avg']))

    def test_already_advanced_scheduler_is_not_advanced_twice(self):
        model, optimizer, scheduler, scaler = self.components()
        restore_training_state(model, optimizer, scheduler, scaler, self.checkpoint, self.history)
        advanced = {**self.checkpoint, 'optimizer': copy.deepcopy(optimizer.state_dict()),
                    'scheduler': copy.deepcopy(scheduler.state_dict())}
        model, optimizer, scheduler, scaler = self.components()
        self.assertEqual(restore_training_state(model, optimizer, scheduler, scaler, advanced, self.history), 34)
        self.assertEqual(scheduler.last_epoch, 33)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .00001875)

    def test_missing_history_epoch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'history'):
            restore_training_state(*self.components(), self.checkpoint, self.history[:-1])

    def test_weight_identity_guard_distinguishes_a_changed_checkpoint(self):
        model = build_model()
        first = state_fingerprint(model.state_dict())
        self.assertEqual(first, state_fingerprint(model.state_dict()))
        with torch.no_grad():
            model.classifier[1].weight[0, 0] += .001
        self.assertNotEqual(first, state_fingerprint(model.state_dict()))

    def test_cached_rgb_decode_matches_original_without_more_source_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'one_class/cell.png'; file.parent.mkdir()
            Image.new('RGB', (64, 64), (40, 100, 150)).save(file)
            raw = file.read_bytes()
            dataset = CachedImageFolder(Path(folder), T.ToTensor(), {str(file): raw})
            file.unlink()  # Temporary fixture only: decoding must now come from the cache.
            tensor, target = dataset[0]
            self.assertEqual(target, 0)
            self.assertTrue(torch.allclose(tensor[:, 0, 0], torch.tensor([40, 100, 150]) / 255))


if __name__ == '__main__':
    unittest.main(verbosity=2)
