"""Shared model, preprocessing, serialization and multiclass metrics."""
import json
from pathlib import Path
import uuid

import numpy as np

CLASSES = ["basophil", "eosinophil", "erythroblast", "immature_granulocytes",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def build_model(pretrained=False):
    import torch.nn as nn
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    return model


def transforms_for(config, training=False):
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    if training:
        steps = [T.RandomResizedCrop(config['image_size'], scale=(.85, 1.),
                                    interpolation=InterpolationMode.BICUBIC),
                 T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
                 T.ColorJitter(brightness=.1, contrast=.1, saturation=.05)]
    else:
        steps = [T.Resize(config['resize_size'], interpolation=InterpolationMode.BICUBIC),
                 T.CenterCrop(config['image_size'])]
    return T.Compose(steps + [T.ToTensor(), T.Normalize(config['mean'], config['std'])])


def classification_metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(labels), len(CLASSES)) or not len(labels):
        raise ValueError('Expected nonempty N × 8 probabilities')
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid class probabilities')
    if (labels < 0).any() or (labels >= len(CLASSES)).any():
        raise ValueError('Invalid class labels')
    predictions = probabilities.argmax(1)
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(cm, (labels, predictions), 1)
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(cm.diagonal(), predicted, out=np.zeros(8), where=predicted != 0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(8), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(8), where=precision + recall != 0)
    normalized = np.divide(cm, support[:, None], out=np.zeros((8, 8)), where=support[:, None] != 0)
    aucs = []
    for i in range(8):
        positive = labels == i
        npos, nneg = int(positive.sum()), int((~positive).sum())
        if not npos or not nneg:
            aucs.append(None)
            continue
        order = np.argsort(-probabilities[:, i], kind='stable')
        scores, pos = probabilities[order, i], positive[order]
        ends = np.r_[np.where(np.diff(scores))[0], len(scores) - 1]
        tpr = np.r_[0., np.cumsum(pos)[ends] / npos]
        fpr = np.r_[0., np.cumsum(~pos)[ends] / nneg]
        aucs.append(float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)))
    return {'accuracy': float(cm.trace() / len(labels)), 'macro_precision': float(precision.mean()),
            'macro_recall': float(recall.mean()), 'macro_f1': float(f1.mean()),
            'weighted_f1': float(np.dot(f1, support) / len(labels)),
            'macro_auroc_ovr': float(np.mean(aucs)) if all(a is not None for a in aucs) else None,
            'auroc_per_class_ovr': dict(zip(CLASSES, aucs)), 'confusion_matrix': cm.tolist(),
            'normalized_confusion_matrix': normalized.tolist(),
            'per_class': [{'class': c, 'precision': float(precision[i]), 'recall': float(recall[i]),
                           'f1': float(f1[i]), 'support': int(support[i])} for i, c in enumerate(CLASSES)]}


"""Final artifacts from one held-out test inference pass."""
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch



def write_curves(history, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = [r['epoch'] for r in history]
    for ax, keys, title in zip(axes, [('train_loss', 'val_loss'), ('train_accuracy', 'val_accuracy'), ('val_macro_f1',)],
                               ['Cross-entropy loss', 'Accuracy', 'Validation macro F1']):
        for key in keys:
            ax.plot(epochs, [r[key] for r in history], label=key)
        ax.set(xlabel='Epoch', title=title); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(Path(out) / 'training_curves.png', dpi=150); plt.close(fig)


def evaluate_once(model, loader, device, output, config, best_epoch, history):
    output = Path(output)
    if (output / 'evaluation.json').exists():
        raise RuntimeError('Final evaluation already exists; refusing another test pass.')
    model.eval()
    # Warm up on a validation-shaped synthetic input, never on a test image.
    with torch.inference_mode():
        model(torch.zeros(1, 3, config['image_size'], config['image_size'], device=device))
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    labels, probabilities = [], []
    forward_seconds = 0.
    synchronize(); started = time.perf_counter()
    with torch.inference_mode():
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            synchronize(); tick = time.perf_counter()
            logits = model(images)
            synchronize(); forward_seconds += time.perf_counter() - tick
            probabilities.extend(logits.softmax(1).cpu().tolist())
            labels.extend(target.tolist())
    synchronize(); elapsed = time.perf_counter() - started
    result = classification_metrics(labels, probabilities)
    result.update({'test_images': len(labels), 'best_epoch': best_epoch, 'device': str(device),
                   'total_inference_seconds': elapsed, 'average_inference_ms': elapsed / len(labels) * 1000,
                   'forward_seconds': forward_seconds, 'forward_average_ms': forward_seconds / len(labels) * 1000,
                   'timing_note': 'FP32 batched GPU inference. Total includes loading/transfers/probabilities; forward excludes warmup and loading. This is not single-image Mac latency.',
                   'classes': CLASSES, 'test_evaluation_passes': 1,
                   'data_policy': config.get('data_policy'), 'manifest_sha256': config.get('manifest_sha256')})
    p = np.asarray(probabilities); y = np.asarray(labels); pred = p.argmax(1)
    ds = loader.dataset
    with (output / 'predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'true_class', 'predicted_class', 'confidence'] + ['p_' + c for c in CLASSES])
        for index, (target, guessed, probs) in enumerate(zip(y, pred, p)):
            writer.writerow([Path(ds.samples[index][0]).name, CLASSES[target], CLASSES[guessed], float(probs[guessed])] + probs.tolist())
    with (output / 'per_class.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['class', 'precision', 'recall', 'f1', 'support'])
        writer.writeheader(); writer.writerows(result['per_class'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key in ('confusion_matrix', 'normalized_confusion_matrix'):
        matrix = np.asarray(result[key])
        fig, ax = plt.subplots(figsize=(10, 9))
        image = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1 if key.startswith('normalized') else None)
        ax.set_xticks(range(8), CLASSES, rotation=45, ha='right'); ax.set_yticks(range(8), CLASSES)
        ax.set(xlabel='Predicted class', ylabel='True class', title=key.replace('_', ' ').title())
        for i in range(8):
            for j in range(8):
                ax.text(j, i, f'{matrix[i,j]:.2f}' if key.startswith('normalized') else str(matrix[i,j]),
                        ha='center', va='center', color='white' if matrix[i,j] > matrix.max() / 2 else 'black')
        fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(output / (key + '.png'), dpi=150); plt.close(fig)
        np.savetxt(output / (key + '.csv'), matrix, delimiter=',', fmt='%.6f' if key.startswith('normalized') else '%d',
                   header=','.join(CLASSES), comments='')
    examples = []
    examples_dir = output / 'examples'; examples_dir.mkdir(exist_ok=True)
    # Deterministic examples: one correct and up to two highest-confidence errors per true class.
    selected = []
    for cls in range(8):
        correct = np.where((y == cls) & (pred == y))[0]
        wrong = np.where((y == cls) & (pred != y))[0]
        if len(correct):
            selected.append(int(correct[0]))
        if len(wrong):
            selected.extend(int(i) for i in wrong[np.argsort(-p[wrong].max(1))[:2]])
    for index in selected:
        name = f'cell_{index:05d}.png'
        with Image.open(ds.samples[index][0]) as im:
            im.convert('RGB').save(examples_dir / name)
        examples.append({'image': 'examples/' + name, 'true_class': CLASSES[y[index]],
                         'predicted_class': CLASSES[pred[index]], 'correct': bool(y[index] == pred[index]),
                         'confidence': float(p[index, pred[index]]), 'probabilities': dict(zip(CLASSES, p[index].tolist()))})
    result['examples'] = examples
    cm = np.asarray(result['confusion_matrix']).copy(); np.fill_diagonal(cm, 0)
    pairs = sorted([(int(cm[i,j]), CLASSES[i], CLASSES[j]) for i in range(8) for j in range(8) if cm[i,j]], reverse=True)
    result['common_errors'] = [{'count': n, 'true_class': a, 'predicted_class': b} for n, a, b in pairs[:12]]
    result['overfitting_warnings'] = [f"Epoch {history[i]['epoch']}: training loss decreased while validation loss increased."
                                      for i in range(1, len(history)) if history[i]['train_loss'] < history[i-1]['train_loss'] and history[i]['val_loss'] > history[i-1]['val_loss']]
    save_json(output / 'evaluation.json', result)
    lines = ['# Final held-out test evaluation', '', 'The checkpoint was selected using validation macro F1, followed by one test inference pass.', '',
             f"Best epoch: {best_epoch}. Test images: {len(labels)}.", '',
             '| Metric | Value |', '|---|---:|']
    lines += [f'| {key} | {result[key]:.6f} |' for key in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1']]
    lines += [f"| Macro one-vs-rest AUROC | {result['macro_auroc_ovr']} |", '',
              '| Class | Precision | Recall | F1 | Support |', '|---|---:|---:|---:|---:|']
    lines += [f"| {r['class']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['support']} |" for r in result['per_class']]
    lines += ['', '![Confusion matrix](confusion_matrix.png)', '![Normalized confusion matrix](normalized_confusion_matrix.png)',
              '![Training history](training_curves.png)', '', '## Interpretation', '',
              'Precision describes reliability of a predicted class; recall describes how many true cells of that class were found. Macro F1 weights all eight classes equally. Confidence is an uncalibrated softmax score, not clinical certainty.',
              'Review immature granulocyte / neutrophil / eosinophil confusions. A low-resolution image may hide morphological detail.',
              'Donor-level independence is unverified. Exact duplicate exclusions do not eliminate near-duplicates or domain shift. One known duplicated image carries conflicting eosinophil/neutrophil labels; loader exclusions do not establish the correct label.',
              f"Data policy: {config.get('data_policy')}. Original labels and full test split were retained.", '', '## Most common errors', '']
    lines += [f"- {r['true_class']} → {r['predicted_class']}: {r['count']}" for r in result['common_errors']]
    lines += ['', '## Timing', '', f"{elapsed:.3f}s total; {elapsed/len(labels)*1000:.3f}ms/image. {result['timing_note']}", '', '## Potential overfitting', '']
    lines += ['- ' + w for w in result['overfitting_warnings']] or ['No epoch-wise warning detected.']
    lines += ['', '## Example predictions', '']
    for ex in examples:
        lines += [f"![{ex['true_class']} predicted {ex['predicted_class']} ({ex['confidence']:.3f})]({ex['image']})"]
    (output / 'evaluation_report.md').write_text('\n'.join(lines) + '\n')
    return result


"""GPU-only transfer learning; Kaggle entry point is generated by manage.py prepare."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder



def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed); random.seed(seed)


def train(data, output, config, manifest, policy, weights_file):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'best.pt').exists() or (output / 'history.json').exists():
        raise RuntimeError('This output already contains training artifacts; use a distinct run directory.')
    startup = {'cuda_available': torch.cuda.is_available(), 'torch_version': str(torch.__version__),
               'torchvision_version': __import__('torchvision').__version__,
               'torch_cuda_build': torch.version.cuda,
               'gpu_count': torch.cuda.device_count(), 'started_at': datetime.now(timezone.utc).isoformat()}
    save_json(output / 'startup.json', startup)
    if not startup['cuda_available']:
        raise RuntimeError('CUDA GPU unavailable. Full CPU training is disabled. Enable a Kaggle GPU accelerator.')
    device = torch.device('cuda:0')
    startup['gpu'] = torch.cuda.get_device_name(device)
    save_json(output / 'startup.json', startup)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    excluded = set(policy['excluded_paths'])
    # Verify hashes on Kaggle against the locally audited manifest before training starts.
    with Path(manifest).open() as f:
        records = list(csv.DictReader(f))
    for row in records:
        p = data / row['path']
        if hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('Dataset hash mismatch: ' + row['path'])
    expected_paths = {r['path'] for r in records}
    actual_paths = {p.relative_to(data).as_posix() for s in ('train', 'val', 'test') for p in (data/s).rglob('*') if p.is_file()}
    if expected_paths != actual_paths:
        raise RuntimeError('Dataset file inventory differs from audited manifest')
    datasets, loaders = {}, {}
    effective = {}
    for split in ('train', 'val'):
        ds = ImageFolder(data / split, transform=transforms_for(config, split == 'train'))
        if ds.classes != CLASSES:
            raise RuntimeError('Class mapping mismatch')
        indices = [i for i, (p, _) in enumerate(ds.samples) if Path(p).relative_to(data).as_posix() not in excluded]
        datasets[split] = Subset(ds, indices)
        effective[split] = len(indices)
        generator = torch.Generator().manual_seed(config['seed'])
        loaders[split] = DataLoader(datasets[split], batch_size=config['batch_size'], shuffle=split == 'train',
                                    num_workers=config['workers'], pin_memory=True, worker_init_fn=seed_worker,
                                    generator=generator, persistent_workers=config['workers'] > 0)
    config.update({'data_policy': policy, 'effective_split_sizes': effective,
                   'manifest_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest()})
    save_json(output / 'model_config.json', config)
    save_json(output / 'class_mapping.json', {str(i): c for i, c in enumerate(CLASSES)})
    # Offline ImageNet initialization uses the complete original 1000-class state dict.
    from torchvision.models import efficientnet_b0
    base = efficientnet_b0(weights=None)
    base.load_state_dict(torch.load(weights_file, map_location='cpu', weights_only=True))
    import torch.nn as nn
    base.classifier[1] = nn.Linear(base.classifier[1].in_features, 8)
    model = base.to(device)
    if not all(p.is_cuda for p in model.parameters()):
        raise RuntimeError('Model parameters are not all on CUDA')
    startup['model_on_cuda'] = True
    config['parameter_count'] = sum(p.numel() for p in model.parameters())
    save_json(output / 'model_config.json', config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    history, best_f1, best_epoch, stale = [], -1., 0, 0
    started = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        lr = optimizer.param_groups[0]['lr']
        metrics = {}
        for split in ('train', 'val'):
            training = split == 'train'; model.train(training)
            loss_sum, correct, count, ys, probs = 0., 0, 0, [], []
            with torch.set_grad_enabled(training):
                for images, labels in loaders[split]:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    if images.device != device or labels.device != device:
                        raise RuntimeError('Training batch is not on requested GPU')
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=training):
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
            metrics[split + '_loss'] = loss_sum / count
            metrics[split + '_accuracy'] = correct / count
            if not training:
                metrics['val_macro_f1'] = classification_metrics(ys, probs)['macro_f1']
        row = {'epoch': epoch, 'planned_epochs': config['epochs'], **metrics, 'learning_rate': lr,
               'elapsed_seconds': time.perf_counter() - started, 'gpu': startup['gpu']}
        history.append(row)
        save_json(output / 'history.json', history)
        with (output / 'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        improved = metrics['val_macro_f1'] > best_f1
        substantial_improvement = metrics['val_macro_f1'] > best_f1 + config['min_delta']
        stale = 0 if substantial_improvement else stale + 1
        if improved:
            best_f1, best_epoch = metrics['val_macro_f1'], epoch
            checkpoint = {'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          'config': config, 'best_epoch': best_epoch, 'validation_macro_f1': best_f1}
            torch.save(checkpoint, output / 'best.tmp'); (output / 'best.tmp').replace(output / 'best.pt')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'config': config}, output / 'last.pt')
        scheduler.step(metrics['val_macro_f1'])
        save_json(output / 'progress.json', {'status': 'running', 'best_epoch': best_epoch, **row})
        print('CELLSCOPE_METRIC ' + json.dumps(row), flush=True)
        if stale >= config['patience']:
            break
    write_curves(history, output)
    checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict']); model.to(device)
    # The test loader is created only after checkpoint selection is final.
    test_ds = ImageFolder(data / 'test', transform=transforms_for(config))
    if test_ds.classes != CLASSES:
        raise RuntimeError('Test class mapping mismatch')
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['workers'], pin_memory=True)
    result = evaluate_once(model, test_loader, device, output, config, best_epoch, history)
    save_json(output / 'progress.json', {'status': 'completed', 'best_epoch': best_epoch, **history[-1]})
    print('CELLSCOPE_COMPLETE ' + json.dumps({k: result[k] for k in ['accuracy', 'macro_f1', 'best_epoch']}), flush=True)


def resolve_kaggle_inputs(input_root, working_root):
    """Kaggle expands uploaded ZIPs under their basename; also support intact archives."""
    manifests = [p for p in Path(input_root).rglob('manifest.csv')
                 if all((p.parent / name).is_file() for name in ['config.json', 'data_policy.json', 'efficientnet_b0_rwightman-7f5810bc.pth'])]
    if len(manifests) != 1:
        raise RuntimeError('Expected exactly one audited BloodMNIST assets manifest')
    assets = manifests[0].parent
    for candidate in (assets / 'bloodmnist', assets):
        if all((candidate / s).is_dir() for s in ('train', 'val', 'test')):
            return assets, candidate
    archive_path = assets / 'bloodmnist.zip'
    if not archive_path.is_file():
        raise RuntimeError('Assets found, but neither expanded BloodMNIST splits nor an intact ZIP exists')
    data = Path(working_root) / 'bloodmnist'; data.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not (data / name).resolve().is_relative_to(data.resolve()):
                raise RuntimeError('Unsafe archive member')
        archive.extractall(data)
    return assets, data


def kaggle_main():
    output = Path('/kaggle/working/cellscope')
    output.mkdir(parents=True, exist_ok=True)
    try:
        diagnostics = {'cuda_available': torch.cuda.is_available(), 'torch_version': str(torch.__version__),
                       'torch_cuda_build': torch.version.cuda, 'gpu_count': torch.cuda.device_count()}
        if diagnostics['cuda_available']:
            diagnostics['gpu'] = torch.cuda.get_device_name(0)
        save_json(output / 'startup.json', diagnostics)
        print('CELLSCOPE_STARTUP ' + json.dumps(diagnostics), flush=True)
        # Fail before extracting assets if Kaggle supplied a CPU-only session.
        if not diagnostics['cuda_available']:
            raise RuntimeError('CUDA unavailable. Set Kaggle Settings > Accelerator > GPU T4 x2, then Save Version > Run All.')
        assets, data = resolve_kaggle_inputs('/kaggle/input', '/kaggle/working')
        config = json.loads((assets / 'config.json').read_text())
        policy = json.loads((assets / 'data_policy.json').read_text())
        train(data, output, config, assets / 'manifest.csv', policy, assets / 'efficientnet_b0_rwightman-7f5810bc.pth')
    except Exception as exc:
        save_json(output / 'progress.json', {'status': 'error', 'error': str(exc), 'type': type(exc).__name__})
        raise


if __name__ == '__main__':
    kaggle_main()
