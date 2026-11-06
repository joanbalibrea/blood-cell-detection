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
