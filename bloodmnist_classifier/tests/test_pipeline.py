"""Small offline checks: real data policy, metric math, GPU guard and inference contract."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'training'))
from common import CLASSES, build_model, classification_metrics, transforms_for
from train import train, resolve_kaggle_inputs
from inference import CellClassifier


class PipelineChecks(unittest.TestCase):
    def test_fragmented_log_records_and_epoch_deduplication_in_temporary_root(self):
        from unittest.mock import patch
        import manage
        row = {'epoch': 1, 'planned_epochs': 40, 'train_loss': 2., 'val_loss': 2.,
               'train_accuracy': .125, 'val_accuracy': .125, 'val_macro_f1': .02,
               'learning_rate': .0003, 'elapsed_seconds': 1., 'gpu': 'synthetic test device'}
        line = 'CELLSCOPE_METRIC ' + json.dumps(row) + '\n'
        class FixtureClient:
            def kernels_logs_stream(self, ref):
                return iter([{'data': line[:25]}, {'data': line[25:]}, {'data': line}])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'outputs').mkdir()
            (root / 'run.json').write_text(json.dumps({'kernel': 'fixture/test', 'version': 1}))
            with patch.object(manage, 'ROOT', root), patch.object(manage, 'RUN', root / 'run.json'):
                manage.stream_logs_once(FixtureClient())
            measured = json.loads((root / 'dashboard/data/history.json').read_text())
            self.assertEqual(measured, [row])

    def test_kaggle_expanded_zip_and_intact_archive_layouts(self):
        import zipfile
        with tempfile.TemporaryDirectory() as folder:
            mount = Path(folder) / 'input/assets'; mount.mkdir(parents=True)
            for name in ['manifest.csv', 'config.json', 'data_policy.json', 'efficientnet_b0_rwightman-7f5810bc.pth']:
                (mount / name).touch()
            for split in ['train', 'val', 'test']:
                (mount / 'bloodmnist' / split).mkdir(parents=True)
            assets, data = resolve_kaggle_inputs(Path(folder) / 'input', Path(folder) / 'working')
            self.assertEqual(assets, mount); self.assertEqual(data, mount / 'bloodmnist')
            import shutil
            shutil.rmtree(mount / 'bloodmnist')
            with zipfile.ZipFile(mount / 'bloodmnist.zip', 'w') as archive:
                for split in ['train', 'val', 'test']:
                    archive.writestr(split + '/cell.txt', 'fixture')
            assets, data = resolve_kaggle_inputs(Path(folder) / 'input', Path(folder) / 'working')
            self.assertTrue((data / 'train/cell.txt').exists())

    def test_metric_math_perfect_and_tied(self):
        truth = np.arange(8)
        perfect = classification_metrics(truth, np.eye(8))
        for key in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1', 'macro_auroc_ovr']:
            self.assertAlmostEqual(perfect[key], 1.)
        tied = classification_metrics(truth, np.ones((8, 8)) / 8)
        self.assertAlmostEqual(tied['accuracy'], 1 / 8)
        self.assertAlmostEqual(tied['macro_auroc_ovr'], .5)
        self.assertAlmostEqual(tied['macro_f1'], (2 / 9) / 8)

    def test_missing_class_auc_is_unavailable(self):
        result = classification_metrics([0], [[1.] + [0.] * 7])
        self.assertIsNone(result['macro_auroc_ovr'])
        self.assertEqual(result['per_class'][1]['recall'], 0.)

    def test_policy_removes_cross_split_pixel_leakage(self):
        import csv
        report = json.loads((ROOT / 'reports/generated/dataset.json').read_text())
        excluded = set(report['recommended_exclusions']['excluded_paths'])
        hashes = {s: set() for s in ['train', 'val', 'test']}
        with (ROOT / 'reports/generated/manifest.csv').open() as file:
            for r in csv.DictReader(file):
                if r['path'] not in excluded:
                    hashes[r['split']].add(r['pixel_sha256'])
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            self.assertFalse(hashes[a] & hashes[b])
        self.assertFalse(any(p.startswith('test/') for p in excluded))

    @unittest.skipIf(torch.cuda.is_available(), 'CPU refusal test applies to non-CUDA local host')
    def test_full_cpu_training_refused_before_reading_data(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'CUDA GPU unavailable'):
                train('/nonexistent', folder, {}, '/nonexistent', {}, '/nonexistent')
            self.assertFalse(json.loads((Path(folder) / 'startup.json').read_text())['cuda_available'])

    def test_local_inference_contract_with_untrained_synthetic_checkpoint(self):
        # This verifies loading/preprocessing/device/probability behavior, never model quality.
        config = json.loads((ROOT / 'training/config.json').read_text())
        torch.set_num_threads(2)
        model = build_model(pretrained=False)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic_untrained.pt'
            torch.save({'state_dict': model.state_dict(), 'config': config}, path)
            predictor = CellClassifier(path, device='cpu')
            result = predictor.predict(Image.new('RGB', (64, 64), (180, 140, 150)))
            self.assertIn(result['predicted_class'], CLASSES)
            self.assertEqual(list(result['probabilities']), CLASSES)
            self.assertAlmostEqual(sum(result['probabilities'].values()), 1., places=5)
            self.assertTrue(0 <= result['confidence'] <= 1)
            self.assertEqual(tuple(transforms_for(config)(Image.new('RGB', (64, 64))).shape), (3, 224, 224))

    def test_final_report_and_once_only_guard_with_synthetic_fixture(self):
        from evaluation import evaluate_once, write_curves
        from torch.utils.data import DataLoader, Dataset
        class FixtureDataset(Dataset):
            def __init__(self, folder):
                self.samples = []
                for i in range(8):
                    path = Path(folder) / f'fixture_{i}.png'
                    Image.new('RGB', (64, 64), (i * 30, 128, 128)).save(path)
                    self.samples.append((str(path), i))

            def __len__(self):
                return 8

            def __getitem__(self, i):
                return torch.zeros(3, 224, 224), i

        class FixtureModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(len(x), 8, device=x.device)

        config = json.loads((ROOT / 'training/config.json').read_text())
        history = [{'epoch': 1, 'train_loss': 2., 'val_loss': 2., 'train_accuracy': .125,
                    'val_accuracy': .125, 'val_macro_f1': .02}]
        with tempfile.TemporaryDirectory() as folder:
            dataset = FixtureDataset(folder); loader = DataLoader(dataset, batch_size=4)
            write_curves(history, folder)
            result = evaluate_once(FixtureModel(), loader, torch.device('cpu'), folder, config, 1, history)
            self.assertEqual(result['test_evaluation_passes'], 1)
            self.assertAlmostEqual(result['accuracy'], .125)
            for name in ['predictions.csv', 'evaluation.json', 'evaluation_report.md', 'per_class.csv',
                         'confusion_matrix.png', 'normalized_confusion_matrix.png', 'training_curves.png']:
                self.assertTrue((Path(folder) / name).is_file())
            with self.assertRaisesRegex(RuntimeError, 'already exists'):
                evaluate_once(FixtureModel(), loader, torch.device('cpu'), folder, config, 1, history)


if __name__ == '__main__':
    unittest.main(verbosity=2)
