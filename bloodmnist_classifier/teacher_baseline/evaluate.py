"""Evaluate the supplied six-class checkpoint without touching eight-class run files."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / 'models/resnet18-bloodmnist-wbc.pt'
DEFAULT_DATA = ROOT.parent.parent / 'bloodmnist'
DEFAULT_REPORT = ROOT / 'reports/generated/teacher_baseline'


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def metrics(y, p, classes):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    n = len(classes)
    if p.shape != (len(y), n) or not len(y) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid probabilities/support')
    predicted = p.argmax(1)
    matrix = np.zeros((n, n), dtype=int)
    np.add.at(matrix, (y, predicted), 1)
    support, pred_support, tp = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    precision = np.divide(tp, pred_support, out=np.zeros(n), where=pred_support != 0)
    recall = np.divide(tp, support, out=np.zeros(n), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n), where=(precision + recall) != 0)
    normalized = np.divide(matrix, support[:, None], out=np.zeros((n, n)), where=support[:, None] != 0)
    aucs = []
    for i in range(n):
        positive = y == i; pos, neg = int(positive.sum()), int((~positive).sum())
        if not pos or not neg:
            aucs.append(None); continue
        order = np.argsort(-p[:, i], kind='stable')
        scores, labels = p[order, i], positive[order]
        ends = np.r_[np.flatnonzero(np.diff(scores)), len(y) - 1]
        tpr, fpr = np.r_[0., np.cumsum(labels)[ends] / pos], np.r_[0., np.cumsum(~labels)[ends] / neg]
        aucs.append(float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)))
    return {'images': len(y), 'accuracy': float(tp.sum() / len(y)), 'macro_precision': float(precision.mean()),
            'macro_recall': float(recall.mean()), 'macro_f1': float(f1.mean()),
            'weighted_f1': float(np.dot(f1, support) / len(y)),
            'macro_auroc_ovr': float(np.mean(aucs)) if all(v is not None for v in aucs) else None,
            'confusion_matrix': matrix.tolist(), 'normalized_confusion_matrix': normalized.tolist(),
            'per_class': [{'class': c, 'precision': float(precision[i]), 'recall': float(recall[i]),
                           'f1': float(f1[i]), 'support': int(support[i])} for i, c in enumerate(classes)]}


class CellImages(Dataset):
    def __init__(self, source, rows, transform):
        self.source, self.rows, self.transform = source, rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.source / row['path']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise RuntimeError('Source differs from audited image: ' + row['path'])
        import io
        with Image.open(io.BytesIO(raw)) as im:
            image = self.transform(im.convert('RGB'))
        return image, row['target']


def infer(model, loader, device):
    y, probabilities = [], []
    forward_seconds = 0.
    def sync():
        if device.type == 'mps':
            torch.mps.synchronize()
    sync(); started = time.perf_counter()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device); sync(); tick = time.perf_counter()
            logits = model(images); sync(); forward_seconds += time.perf_counter() - tick
            probabilities.extend(logits.softmax(1).cpu().tolist()); y.extend(labels.tolist())
    sync()
    return np.asarray(y), np.asarray(probabilities), {'total_seconds': time.perf_counter() - started,
                                                    'forward_seconds': forward_seconds}


def matrix_plot(matrix, classes, path, normalized=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    array = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(array, cmap='Blues', vmin=0, vmax=1 if normalized else None)
    ax.set_xticks(range(6), [c.replace('_', '\n') for c in classes], rotation=30, ha='right')
    ax.set_yticks(range(6), [c.replace('_', ' ') for c in classes])
    ax.set(xlabel='Predicted', ylabel='True', title='Teacher ResNet18 — ' + ('row-normalized' if normalized else 'count') + ' confusion matrix')
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{array[i,j]:.2f}' if normalized else str(array[i,j]), ha='center', va='center',
                    color='white' if array[i,j] > array.max() / 2 else 'black')
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def evaluate(checkpoint, source, report, device='cpu', batch_size=32):
    checkpoint, source, report = Path(checkpoint).resolve(), Path(source).resolve(), Path(report).resolve()
    report.mkdir(parents=True, exist_ok=True)
    if (report / 'evaluation.json').exists() or (report / 'evaluation_started.json').exists():
        raise RuntimeError('A baseline evaluation was already started here. Inspect its artifacts; implicit repeated test evaluation is disabled.')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    required = ['architecture', 'classes', 'input_size', 'mean', 'std', 'state_dict']
    if not isinstance(payload, dict) or not all(k in payload for k in required):
        raise ValueError('Checkpoint lacks required architecture/classes/preprocessing metadata')
    classes = payload['classes']
    expected = {'basophil', 'eosinophil', 'immature_granulocytes', 'lymphocyte', 'monocyte', 'neutrophil'}
    if payload['architecture'] != 'resnet18' or len(classes) != 6 or set(classes) != expected:
        raise ValueError('Expected the supplied six-class WBC ResNet18 checkpoint')
    if tuple(payload['state_dict']['fc.weight'].shape) != (6, 512):
        raise ValueError('Checkpoint head does not match declared six classes')
    config = {'architecture': payload['architecture'], 'classes': classes, 'input_size': payload['input_size'],
              'mean': payload['mean'], 'std': payload['std'],
              'resize': 'direct square bilinear, antialias=True',
              'preprocessing_assumption': 'Checkpoint supplies input size and normalization but no interpolation/crop recipe. Direct square bilinear resize is an explicit assumption; exact teacher reproduction is not established.'}
    transform = T.Compose([T.Resize((config['input_size'], config['input_size']), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                           T.ToTensor(), T.Normalize(config['mean'], config['std'])])
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; use CPU')
    device = torch.device(device)
    torch.set_num_threads(2)
    model = resnet18(weights=None); model.fc = torch.nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(payload['state_dict'], strict=True); model.to(device).eval()
    audit = json.loads((ROOT / 'reports/generated/dataset.json').read_text())
    manifest_path = ROOT / 'reports/generated/manifest.csv'
    with manifest_path.open() as file:
        manifest = list(csv.DictReader(file))
    if source != Path(audit['source']).resolve():
        raise ValueError('Requested dataset differs from the audited source')
    # Same loader exclusion policy for validation, without touching the eight-class files.
    excluded_val = set(audit['recommended_exclusions']['excluded_paths'])
    conflicts = {p for group in audit.get('conflicting_label_duplicate_groups', []) for p in group}
    rows_by_split = {split: [{**r, 'target': classes.index(r['class'])} for r in manifest
                            if r['split'] == split and r['class'] in classes
                            and not (split == 'val' and r['path'] in excluded_val)] for split in ['val', 'test']}
    info = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'checkpoint_path': str(checkpoint),
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'architecture': 'resnet18', 'classes': classes, 'config': config,
            'checkpoint_epoch': payload.get('epoch'), 'checkpoint_manifest_sha256': payload.get('manifest_sha256'),
            'local_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'checkpoint_reported_validation': payload.get('validation'),
            'checkpoint_reported_environment': payload.get('environment'),
            'checkpoint_reported_training_config': payload.get('config'),
            'provenance_note': 'Checkpoint metadata is supplied provenance, not independently verified training history. Its manifest hash differs from the local eight-class manifest; equivalence of donors/splits/training data is unverified.',
            'excluded_categories': ['erythroblast', 'platelet'], 'device': str(device),
            'torch': str(torch.__version__), 'torchvision': __import__('torchvision').__version__,
            'test_passes': 1, 'parameter_count': sum(p.numel() for p in model.parameters())}
    save(report / 'evaluation_started.json', {'started_at': info['evaluated_at'], 'checkpoint_sha256': info['checkpoint_sha256'], 'config': config})
    # Warmup is synthetic, never a held-out image.
    with torch.inference_mode():
        model(torch.zeros(1, 3, config['input_size'], config['input_size'], device=device))
    for split in ['val', 'test']:
        rows = rows_by_split[split]
        loader = DataLoader(CellImages(source, rows, transform), batch_size=batch_size, shuffle=False, num_workers=0)
        y, p, timing = infer(model, loader, device)
        measured = metrics(y, p, classes)
        measured['timing'] = {**timing, 'average_ms_per_image': 1000 * timing['total_seconds'] / len(rows),
                              'note': 'Batched FP32 local inference; total includes file hashing, decoding, preprocessing, transfer and probabilities. Excludes model loading/warmup. Not single-image latency.'}
        info['validation' if split == 'val' else 'test'] = measured
        with (report / (split + '_predictions.csv')).open('w', newline='') as file:
            writer = csv.writer(file); writer.writerow(['path', 'true_class', 'predicted_class', 'confidence'] + ['p_' + c for c in classes])
            for row, probs in zip(rows, p):
                index = int(probs.argmax()); writer.writerow([row['path'], row['class'], classes[index], float(probs[index])] + probs.tolist())
        print(f'{split}: {len(rows)} images, accuracy={measured["accuracy"]:.6f}, macro_f1={measured["macro_f1"]:.6f}', flush=True)
        if split == 'test':
            keep = np.array([r['path'] not in conflicts for r in rows])
            info['conflict_quarantined_test_sensitivity'] = metrics(y[keep], p[keep], classes)
            info['conflict_quarantined_test_sensitivity']['excluded_paths'] = [r['path'] for r in rows if r['path'] in conflicts]
            matrix_plot(measured['confusion_matrix'], classes, report / 'confusion_matrix.png')
            matrix_plot(measured['normalized_confusion_matrix'], classes, report / 'normalized_confusion_matrix.png', normalized=True)
            examples = []; folder = report / 'examples'; folder.mkdir(exist_ok=True)
            prediction = p.argmax(1)
            selected = []
            for i in range(6):
                correct = np.flatnonzero((y == i) & (prediction == y))
                incorrect = np.flatnonzero((y == i) & (prediction != y))
                selected.extend(correct[:1].tolist())
                if len(incorrect):
                    selected.extend(incorrect[np.argsort(-p[incorrect].max(1))[:2]].tolist())
            for index in selected:
                name = f'cell_{index:04d}.png'
                with Image.open(source / rows[index]['path']) as im:
                    im.convert('RGB').save(folder / name)
                examples.append({'image': 'examples/' + name, 'true_class': classes[y[index]],
                                 'predicted_class': classes[prediction[index]], 'confidence': float(p[index].max()),
                                 'correct': bool(y[index] == prediction[index])})
            info['examples'] = examples
    reported = info.get('checkpoint_reported_validation') or {}
    measured_validation = info['validation']
    info['validation_reproduction_check'] = {
        'accuracy_absolute_difference': abs(measured_validation['accuracy'] - reported['accuracy']) if 'accuracy' in reported else None,
        'macro_f1_absolute_difference': abs(measured_validation['macro_f1'] - reported['macro_f1']) if 'macro_f1' in reported else None,
        'confusion_matrix_matches': measured_validation['confusion_matrix'] == reported.get('confusion_matrix'),
        'support_matches': [r['support'] for r in measured_validation['per_class']] == reported.get('support'),
        'interpretation': 'Matching stored validation outputs supports the chosen preprocessing on this validation population; it does not establish training-data/donor independence or uniquely identify the teacher transform.'}
    with (report / 'per_class.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['class', 'precision', 'recall', 'f1', 'support'])
        writer.writeheader(); writer.writerows(info['test']['per_class'])
    for name in ['confusion_matrix', 'normalized_confusion_matrix']:
        np.savetxt(report / (name + '.csv'), info['test'][name], delimiter=',', header=','.join(classes), comments='', fmt='%.6f')
    save(report / 'evaluation.json', info)
    write_reports(info, report)
    print('Saved separate six-class baseline reports:', report, flush=True)
    return info


def write_reports(info, report):
    test, val = info['test'], info['validation']
    rows = ['# Teacher ResNet18 — separate six-class baseline', '',
            'This is a fixed pretrained checkpoint evaluated independently. It is not the eight-class Kaggle model.', '',
            f"Checkpoint SHA-256: `{info['checkpoint_sha256']}`. Epoch stored in checkpoint: {info['checkpoint_epoch']}.",
            f"Input: {info['config']['input_size']} × {info['config']['input_size']}; device: {info['device']}.",
            info['config']['preprocessing_assumption'], '',
            f"Measured validation: {val['images']} images, accuracy {val['accuracy']:.6f}, macro F1 {val['macro_f1']:.6f}.",
            'Validation was measured once to audit the supplied checkpoint; no weights, thresholds or preprocessing were tuned using test outcomes.', '',
            f"Held-out six-class test subset: {test['images']} images, one inference pass. Erythroblasts and platelets are excluded because this head cannot predict them.", '',
            '| Test metric | Measured value |', '|---|---:|']
    keys = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1', 'macro_auroc_ovr']
    rows += [f'| {k} | {test[k]} |' for k in keys]
    rows += ['', '| Class | Precision | Recall | F1 | Support |', '|---|---:|---:|---:|---:|']
    rows += [f"| {r['class']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['support']} |" for r in test['per_class']]
    sensitivity = info['conflict_quarantined_test_sensitivity']
    rows += ['', '![Count confusion matrix](confusion_matrix.png)', '![Normalized confusion matrix](normalized_confusion_matrix.png)', '',
             '## Label-conflict sensitivity', '',
             f"The same saved probabilities give accuracy {sensitivity['accuracy']:.6f} and macro F1 {sensitivity['macro_f1']:.6f} after excluding {len(sensitivity['excluded_paths'])} known conflicting-label test image(s); n={sensitivity['images']}. No second inference pass was used.", '',
             '## Interpretation limits', '',
             info['provenance_note'],
             'Validation reproduction check: ' + json.dumps(info.get('validation_reproduction_check', {})),
             'Checkpoint-reported validation metrics are stored separately in JSON and are not presented as newly measured test results. Donor-level independence and original teacher preprocessing are unverified.',
             'Six-class closed-set performance cannot be compared directly with the overall eight-class classifier score: the category set and test denominator differ. The baseline cannot recognize platelets or erythroblasts. Confidence is an uncalibrated softmax score.', '',
             '## Inference timing', '', f"Total {test['timing']['total_seconds']:.3f}s; average {test['timing']['average_ms_per_image']:.3f}ms/image. {test['timing']['note']}", '',
             '## Selected examples', '']
    rows += [f"![{r['true_class']} → {r['predicted_class']}; confidence {r['confidence']:.3f}]({r['image']})" for r in info['examples']]
    (report / 'evaluation_report.md').write_text('\n'.join(rows) + '\n')
    e = html.escape
    cards = ''.join(f'<article><small>{e(k.replace("_", " ").upper())}</small><strong>{test[k]*100:.2f}%</strong></article>' for k in keys[:4])
    table = ''.join(f'<tr><td>{e(r["class"].replace("_", " "))}</td><td>{r["precision"]:.4f}</td><td>{r["recall"]:.4f}</td><td>{r["f1"]:.4f}</td><td>{r["support"]}</td></tr>' for r in test['per_class'])
    examples = ''.join(f'<article><img src="{e(r["image"])}" alt="{e(r["true_class"])}"><p>{e(r["true_class"].replace("_", " "))}<br>→ {e(r["predicted_class"].replace("_", " "))}<br>Confidence {r["confidence"]:.3f}</p></article>' for r in info['examples'])
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CellScope — Teacher six-class baseline</title><style>
    :root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#0d131c;color:#e8eef7;font:15px/1.6 system-ui}}main{{max-width:1140px;margin:auto;padding:32px 24px}}h1{{font-size:38px;line-height:1.2;letter-spacing:-1px}}h1 span,a{{color:#6ce3c4}}p,small{{color:#9aaec5}}.tag{{letter-spacing:2px;color:#b39bff;font-size:11px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}article,.panel{{border:1px solid #283547;background:#151f2c;border-radius:10px;padding:20px}}strong{{display:block;font-size:29px;margin-top:8px}}small{{font-size:10px}}.panel{{margin-top:18px}}.notice{{border-left:3px solid #b39bff;padding:15px;background:#1c2437}}table{{width:100%;border-collapse:collapse;min-width:550px}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #283547;font-size:12px}}.scroll{{overflow:auto}}.matrices{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}img{{width:100%;height:auto}}.examples{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px}}.examples p{{font-size:11px}}code{{overflow-wrap:anywhere}}@media(max-width:700px){{.cards{{grid-template-columns:1fr 1fr}}.matrices{{grid-template-columns:1fr}}h1{{font-size:30px}}main{{padding:22px 14px}}}}
    </style><main><div class="tag">CELLSCOPE / INDEPENDENT BASELINE</div><h1>Teacher ResNet18<br><span>Six-class evaluation</span></h1><p>Fixed supplied checkpoint · {test['images']:,} test images · {e(info['device'])} · input 128 × 128</p><p><a href="/dashboard/">Eight-class dashboard ↗</a> · <a href="evaluation.json">Measured JSON</a> · <a href="test_predictions.csv">All test probabilities</a> · <a href="evaluation_report.md">Detailed report</a></p><div class="notice">This baseline recognizes six leukocyte categories. It cannot classify platelets or erythroblasts. Its score uses a different task and denominator from the eight-class model.</div><p>{e(info['config']['preprocessing_assumption'])}</p><div class="cards">{cards}</div><section class="panel"><h2>Per-class test performance</h2><div class="scroll"><table><thead><tr><th>Category</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{table}</tbody></table></div><p>Weighted F1: {test['weighted_f1']:.6f} · Macro one-vs-rest AUROC: {test['macro_auroc_ovr']}</p></section><section class="panel"><h2>Confusion matrices</h2><div class="matrices"><img src="confusion_matrix.png" alt="Six-class count confusion matrix"><img src="normalized_confusion_matrix.png" alt="Six-class row-normalized confusion matrix"></div></section><section class="panel"><h2>Validation and label sensitivity</h2><p>Measured validation: {val['images']:,} images · accuracy {val['accuracy']:.6f} · macro F1 {val['macro_f1']:.6f}.</p><p>Known label-conflict quarantine: {sensitivity['images']:,} test images · accuracy {sensitivity['accuracy']:.6f} · macro F1 {sensitivity['macro_f1']:.6f}. Computed from the same inference pass.</p></section><section class="panel"><h2>Inference timing</h2><p>{test['timing']['total_seconds']:.2f}s total · {test['timing']['average_ms_per_image']:.2f}ms/image. {e(test['timing']['note'])}</p></section><section class="panel"><h2>Selected correct and incorrect predictions</h2><p>One correct and up to two highest-confidence errors per true class, where available. Selected diagnostic examples, not a random sample.</p><div class="examples">{examples}</div></section><section class="panel"><h2>Provenance and limitations</h2><p>{e(info['provenance_note'])}</p><p>Donor independence and the teacher's exact resize recipe are unverified. Softmax confidence is uncalibrated. No eight-class files or Kaggle training state were modified.</p><p>Checkpoint SHA-256: <code>{e(info['checkpoint_sha256'])}</code></p></section></main></html>'''
    match = info.get('validation_reproduction_check', {})
    if match.get('confusion_matrix_matches') and match.get('support_matches'):
        document = document.replace('<h2>Validation and label sensitivity</h2>', '<h2>Validation and label sensitivity</h2><p>The measured validation confusion matrix and class support exactly match the values stored in the checkpoint. The measured validation accuracy and macro F1 also match; this supports the preprocessing assumption without proving donor independence.</p>')
    (report / 'index.html').write_text(document)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data, args.report, args.device, args.batch_size)
