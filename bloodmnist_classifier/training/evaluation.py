"""Final artifacts from one held-out test inference pass."""
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from common import CLASSES, classification_metrics, save_json


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
