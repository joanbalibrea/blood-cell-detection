"""Continue the completed 33-epoch run to exactly 40, using saved training state."""
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from common import CLASSES, build_model, transforms_for, classification_metrics, save_json
from evaluation import evaluate_once, write_curves


def restore_training_state(model, optimizer, scheduler, scaler, checkpoint, history):
    epoch = int(checkpoint['epoch'])
    if [int(r['epoch']) for r in history] != list(range(1, epoch + 1)):
        raise ValueError('Saved history must be complete and end at the saved checkpoint epoch')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    # Original last.pt was saved immediately BEFORE this epoch's scheduler.step.
    if scheduler.last_epoch == epoch - 1:
        scheduler.step(history[-1]['val_macro_f1'])
    elif scheduler.last_epoch != epoch:
        raise ValueError('Unexpected scheduler epoch; refusing an ambiguous continuation')
    return epoch + 1


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed); random.seed(seed)


class CachedImageFolder(ImageFolder):
    def __init__(self, root, transform, cache):
        self.cache = cache
        super().__init__(root, transform=transform, loader=self.cached_loader)

    def cached_loader(self, path):
        with Image.open(io.BytesIO(self.cache[str(Path(path))])) as image:
            return image.convert('RGB')


def state_fingerprint(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def resume_to_40(data, assets, previous, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'history.json').exists():
        raise RuntimeError('Continuation output already exists; refusing duplicate training')
    startup = {'cuda_available': torch.cuda.is_available(), 'torch_version': str(torch.__version__),
               'torchvision_version': __import__('torchvision').__version__, 'torch_cuda_build': torch.version.cuda,
               'gpu_count': torch.cuda.device_count(), 'started_at': datetime.now(timezone.utc).isoformat()}
    if startup['cuda_available']:
        startup['gpu'] = torch.cuda.get_device_name(0)
    save_json(output / 'startup.json', startup)
    print('CELLSCOPE_STARTUP ' + json.dumps(startup), flush=True)
    if not startup['cuda_available']:
        raise RuntimeError('CUDA unavailable. No CPU training is allowed.')
    checkpoint = torch.load(previous / 'last.pt', map_location='cpu', weights_only=True)
    original_best = torch.load(previous / 'best.pt', map_location='cpu', weights_only=True)
    history = json.loads((previous / 'history.json').read_text())
    config = dict(checkpoint['config'])
    if config['classes'] != CLASSES or config['architecture'] != 'efficientnet_b0' or checkpoint['epoch'] >= 40:
        raise ValueError('Expected a compatible eight-class checkpoint below epoch 40')
    manifest = assets / 'manifest.csv'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != config['manifest_sha256']:
        raise ValueError('Continuation dataset manifest differs from original training')
    with manifest.open() as file:
        records = list(csv.DictReader(file))
    def read_verified(row):
        path = data / row['path']; raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Source checksum mismatch: ' + row['path'])
        return str(path), raw
    # Verify all original bytes once, concurrently. Cache encoded images in RAM to
    # avoid repeating slow remote small-file reads in each subsequent epoch.
    print('CELLSCOPE_PHASE Verifying source bytes and warming the image cache', flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        cache = dict(pool.map(read_verified, records))
    expected = {r['path'] for r in records}
    actual = {p.relative_to(data).as_posix() for split in ['train', 'val', 'test'] for p in (data / split).rglob('*') if p.is_file()}
    if expected != actual:
        raise ValueError('Dataset inventory differs from original audit')
    device = torch.device('cuda:0')
    seed = int(config['seed']) + int(checkpoint['epoch'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    config['epochs'] = 40
    config['continuation'] = {'source': 'joanbalibrearull/cellscope-bloodmnist-classifier/2',
                              'resumed_after_epoch': int(checkpoint['epoch']), 'target_epoch': 40,
                              'early_stopping_disabled': True, 'seed': seed,
                              'rng_note': 'Original checkpoint did not save RNG/worker states; model, optimizer, scheduler and scaler are restored, but bitwise identity with uninterrupted training is not claimed.'}
    model = build_model(pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=3)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    first_epoch = restore_training_state(model, optimizer, scheduler, scaler, checkpoint, history)
    if not all(p.is_cuda for p in model.parameters()):
        raise RuntimeError('Model parameters are not on CUDA')
    startup.update({'model_on_cuda': True, 'resumed_after_epoch': first_epoch - 1})
    save_json(output / 'model_config.json', config)
    save_json(output / 'class_mapping.json', {str(i): c for i, c in enumerate(CLASSES)})
    save_json(output / 'resume_provenance.json', config['continuation'])
    excluded = set(config['data_policy']['excluded_paths'])
    loaders = {}
    for split in ['train', 'val']:
        dataset = CachedImageFolder(data / split, transforms_for(config, split == 'train'), cache)
        if dataset.classes != CLASSES:
            raise ValueError('Class mapping mismatch')
        indices = [i for i, (path, _) in enumerate(dataset.samples) if Path(path).relative_to(data).as_posix() not in excluded]
        generator = torch.Generator().manual_seed(seed)
        loaders[split] = DataLoader(Subset(dataset, indices), batch_size=config['batch_size'], shuffle=split == 'train',
                                    num_workers=config['workers'], pin_memory=True, worker_init_fn=seed_worker,
                                    generator=generator, persistent_workers=config['workers'] > 0)
    shutil.copy2(previous / 'best.pt', output / 'best.pt')
    best_f1, best_epoch = float(original_best['validation_macro_f1']), int(original_best['best_epoch'])
    original_fingerprint = state_fingerprint(original_best['state_dict'])
    previous_elapsed = float(history[-1]['elapsed_seconds'])
    criterion = torch.nn.CrossEntropyLoss()
    started = time.perf_counter()
    save_json(output / 'history.json', history)
    print(f'CELLSCOPE_PHASE Restored epoch {first_epoch-1}; next LR={optimizer.param_groups[0]["lr"]}; training to 40 without early stopping', flush=True)
    for epoch in range(first_epoch, 41):
        lr = optimizer.param_groups[0]['lr']; measured = {}
        for split in ['train', 'val']:
            training = split == 'train'; model.train(training)
            loss_sum, correct, count, ys, probs = 0., 0, 0, [], []
            with torch.set_grad_enabled(training):
                for images, labels in loaders[split]:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    if images.device != device or labels.device != device:
                        raise RuntimeError('Batch is not on CUDA')
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.float16, enabled=training):
                        logits = model(images); loss = criterion(logits, labels)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss')
                    if not startup.get('gpu_execution_verified'):
                        startup.update({'batch_on_cuda': True, 'gpu_execution_verified': True})
                        save_json(output / 'startup.json', startup)
                        print('CELLSCOPE_STARTUP ' + json.dumps(startup), flush=True)
                    if training:
                        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                    loss_sum += float(loss.detach()) * len(labels)
                    correct += int((logits.argmax(1) == labels).sum()); count += len(labels)
                    if not training:
                        ys.extend(labels.cpu().tolist()); probs.extend(logits.float().softmax(1).detach().cpu().tolist())
            measured[split + '_loss'] = loss_sum / count
            measured[split + '_accuracy'] = correct / count
            if not training:
                measured['val_macro_f1'] = classification_metrics(ys, probs)['macro_f1']
        row = {'epoch': epoch, 'planned_epochs': 40, **measured, 'learning_rate': lr,
               'elapsed_seconds': previous_elapsed + time.perf_counter() - started, 'gpu': startup['gpu']}
        history.append(row)
        if measured['val_macro_f1'] > best_f1:
            best_f1, best_epoch = measured['val_macro_f1'], epoch
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'config': config, 'best_epoch': best_epoch, 'validation_macro_f1': best_f1}, output / 'best.pt')
        scheduler.step(measured['val_macro_f1'])
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'config': config,
                    'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()}, output / 'last.pt')
        save_json(output / 'history.json', history)
        with (output / 'history.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        save_json(output / 'progress.json', {'status': 'running', 'best_epoch': best_epoch, **row})
        print('CELLSCOPE_METRIC ' + json.dumps(row), flush=True)
    write_curves(history, output)
    selected = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    if state_fingerprint(selected['state_dict']) == original_fingerprint:
        # Identical selected weights need no second inference pass on the test set.
        for name in ['evaluation.json', 'evaluation_report.md', 'predictions.csv', 'per_class.csv',
                     'confusion_matrix.png', 'normalized_confusion_matrix.png', 'confusion_matrix.csv', 'normalized_confusion_matrix.csv']:
            shutil.copy2(previous / name, output / name)
        shutil.copytree(previous / 'examples', output / 'examples', dirs_exist_ok=True)
        result = json.loads((output / 'evaluation.json').read_text())
        result.update({'evaluation_reused_identical_weights': True, 'completed_training_epochs': 40,
                       'test_evaluation_passes_across_versions': 1, 'selected_state_sha256': original_fingerprint})
        explanation = 'Completed 40 training epochs. Validation-selected weights are identical to the previously evaluated checkpoint; saved test predictions were reused without another test inference pass.'
    else:
        model.load_state_dict(selected['state_dict']); model.to(device).eval()
        dataset = CachedImageFolder(data / 'test', transforms_for(config), cache)
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['workers'], pin_memory=True)
        result = evaluate_once(model, loader, device, output, config, best_epoch, history)
        result.update({'evaluation_reused_identical_weights': False, 'completed_training_epochs': 40,
                       'test_evaluation_passes': 2, 'test_evaluation_passes_across_versions': 2,
                       'test_passes_for_final_checkpoint': 1, 'selected_state_sha256': state_fingerprint(selected['state_dict'])})
        explanation = 'The original checkpoint was already evaluated on this test set. The user requested completion of the fixed 40-epoch budget; model selection still used validation only. This is one test pass for the new selected checkpoint, two across versions, and is not a fresh independent holdout.'
    result['continuation_evaluation_note'] = explanation
    result['overfitting_warnings'] = [f"Epoch {history[i]['epoch']}: training loss decreased while validation loss increased."
                                     for i in range(1, len(history)) if history[i]['train_loss'] < history[i-1]['train_loss'] and history[i]['val_loss'] > history[i-1]['val_loss']]
    save_json(output / 'evaluation.json', result)
    report = output / 'evaluation_report.md'
    report.write_text('# Continuation to 40 epochs\n\n' + explanation + '\n\n' + report.read_text())
    save_json(output / 'progress.json', {'status': 'completed', 'best_epoch': best_epoch, **history[-1]})
    print('CELLSCOPE_COMPLETE ' + json.dumps({'epoch': 40, 'best_epoch': best_epoch}), flush=True)


def continuation_main():
    output = Path('/kaggle/working/cellscope_resume')
    output.mkdir(parents=True, exist_ok=True)
    try:
        if not torch.cuda.is_available():
            save_json(output / 'startup.json', {'cuda_available': False})
            raise RuntimeError('CUDA is unavailable; continuation cannot run on CPU')
        assets, data = resolve_kaggle_inputs('/kaggle/input', '/kaggle/working')
        candidates = [p.parent for p in Path('/kaggle/input').rglob('last.pt')
                      if all((p.parent / n).is_file() for n in ['best.pt', 'history.json', 'evaluation.json'])]
        if len(candidates) != 1:
            raise RuntimeError('Attach the completed original notebook version 2 as a kernel input')
        resume_to_40(data, assets, candidates[0], output)
    except Exception as exc:
        save_json(output / 'progress.json', {'status': 'error', 'error': str(exc), 'type': type(exc).__name__})
        raise


if __name__ == '__main__':
    continuation_main()
