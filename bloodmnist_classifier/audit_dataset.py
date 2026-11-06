"""Read every source file; preserve splits; audit bytes and decoded RGB pixels."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from training.common import CLASSES, save_json

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent.parent / 'bloodmnist'


def audit(source):
    source = Path(source).resolve()
    out = ROOT / 'reports/generated'
    out.mkdir(parents=True, exist_ok=True)
    counts, dimensions, formats, modes = {}, Counter(), Counter(), Counter()
    corrupt, unexpected, records, links = [], [], [], []
    bytes_seen, pixels_seen = defaultdict(list), defaultdict(list)
    sheet = Image.new('RGB', (8 * 160, 195), '#151f2c')
    draw = ImageDraw.Draw(sheet)
    for split in ('train', 'val', 'test'):
        folder = source / split
        if not folder.is_dir():
            raise ValueError(f'Missing split {folder}')
        actual = {p.name for p in folder.iterdir() if p.is_dir()}
        if actual != set(CLASSES):
            raise ValueError(f'Class folders mismatch in {split}: {actual}')
        counts[split] = dict.fromkeys(CLASSES, 0)
        for index, name in enumerate(CLASSES):
            paths = sorted((folder / name).rglob('*'))
            example = False
            for p in paths:
                if not p.is_file():
                    continue
                relative = p.relative_to(source).as_posix()
                if p.is_symlink():
                    links.append(relative)
                if p.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}:
                    unexpected.append(relative)
                    continue
                try:
                    raw = p.read_bytes()
                    with Image.open(p) as im:
                        im.verify()
                    with Image.open(p) as im:
                        im.load()
                        dims, fmt, mode = im.size, im.format, im.mode
                        rgb = im.convert('RGB')
                    byte_hash = hashlib.sha256(raw).hexdigest()
                    pixel_hash = hashlib.sha256(str(dims).encode() + rgb.tobytes()).hexdigest()
                    bytes_seen[byte_hash].append(relative)
                    pixels_seen[pixel_hash].append(relative)
                    counts[split][name] += 1
                    dimensions[f'{dims[0]}x{dims[1]}'] += 1
                    formats[fmt] += 1
                    modes[mode] += 1
                    records.append({'path': relative, 'split': split, 'class': name,
                                    'width': dims[0], 'height': dims[1], 'format': fmt,
                                    'mode': mode, 'sha256': byte_hash, 'pixel_sha256': pixel_hash})
                    if split == 'train' and not example:
                        thumb = rgb.resize((144, 144))
                        sheet.paste(thumb, (index * 160 + 8, 8))
                        draw.text((index * 160 + 8, 160), name.replace('_', '\n'), fill='white')
                        example = True
                except Exception as exc:
                    corrupt.append({'path': relative, 'error': type(exc).__name__})
    def duplicate_groups(seen, across=False):
        return [v for v in seen.values() if len(v) > 1 and (not across or len({p.split('/')[0] for p in v}) > 1)]
    totals = {c: sum(v[c] for v in counts.values()) for c in CLASSES}
    train_counts = list(counts['train'].values())
    result = {'audited_at': datetime.now(timezone.utc).isoformat(), 'source': str(source),
              'classes': CLASSES, 'total_images': len(records), 'counts': counts,
              'split_totals': {s: sum(v.values()) for s, v in counts.items()},
              'class_totals': totals, 'dimensions': dict(dimensions), 'formats': dict(formats),
              'modes': dict(modes), 'corrupt_files': corrupt, 'unexpected_files': unexpected,
              'symlinks': links, 'byte_duplicates': duplicate_groups(bytes_seen),
              'pixel_duplicates': duplicate_groups(pixels_seen),
              'cross_split_byte_duplicates': duplicate_groups(bytes_seen, True),
              'cross_split_pixel_duplicates': duplicate_groups(pixels_seen, True),
              'train_imbalance_ratio': max(train_counts) / min(train_counts) if min(train_counts) else None,
              'limitations': ['Patient/donor IDs are unavailable: patient-level independence cannot be verified.',
                             'Exact-duplicate checks cannot exclude related fields, near-duplicates, or shared donors.',
                             'Resizing 64px images cannot recover missing cellular detail.',
                             'Labels include platelets and erythroblasts, not only mature leukocytes.',
                             'Research dataset performance does not establish clinical validity or performance on YOLO crops.']}
    result['integrity_passed'] = not (corrupt or unexpected or links or any(n == 0 for v in counts.values() for n in v.values()))
    result['ready_for_training'] = result['integrity_passed']
    # Keep test intact. Earlier splits lose only overlapping loader entries; source files never change.
    excluded = []
    for group in duplicate_groups(pixels_seen, True):
        splits = {p.split('/')[0] for p in group}
        keep = 'test' if 'test' in splits else 'val'
        excluded.extend(p for p in group if p.split('/')[0] != keep)
    policy = {'name': 'exclude_cross_split_overlap', 'excluded_paths': sorted(set(excluded)),
              'keep_test_intact': True, 'source_files_modified': False,
              'reason': 'Prevent exact pixel leakage; original split assignments and source files remain unchanged.'}
    result['recommended_exclusions'] = policy
    result['conflicting_label_duplicate_groups'] = [g for g in duplicate_groups(pixels_seen) if len({p.split('/')[1] for p in g}) > 1]
    save_json(out / 'data_policy.json', policy)
    sheet.save(out / 'representative_cells.png')
    with (out / 'manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fingerprint = hashlib.sha256((out / 'manifest.csv').read_bytes()).hexdigest()
    result['manifest_sha256'] = fingerprint
    save_json(out / 'dataset.json', result)
    save_json(ROOT / 'dashboard/data/dataset.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(8)
    for split, color in zip(counts, ['#41b99d', '#9c83ed', '#e4aa64']):
        values = np.array(list(counts[split].values()))
        ax.bar(np.arange(8), values, bottom=bottom, label=split, color=color)
        bottom += values
    ax.set_xticks(np.arange(8), [c.replace('_', '\n') for c in CLASSES], rotation=25, ha='right')
    ax.set_ylabel('Images'); ax.set_title('BloodMNIST class and split distribution'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'class_distribution.png', dpi=150); plt.close(fig)
    rows = ['# BloodMNIST dataset audit', '', f"Verified readable images: **{len(records):,}**. Splits were preserved.", '',
            '| Class | Train | Validation | Test | Total |', '|---|---:|---:|---:|---:|']
    rows += [f"| {c} | {counts['train'][c]} | {counts['val'][c]} | {counts['test'][c]} | {totals[c]} |" for c in CLASSES]
    rows += ['', f'Dimensions: {dict(dimensions)}. Formats: {dict(formats)}. Modes: {dict(modes)}.',
             f'Corrupt files: {len(corrupt)}. Unexpected files: {len(unexpected)}. Symlinks: {len(links)}.',
             f"Cross-split duplicate groups: bytes={len(result['cross_split_byte_duplicates'])}, decoded pixels={len(result['cross_split_pixel_duplicates'])}.",
             f"Recommended loader exclusions: {len(policy['excluded_paths'])}. See [data policy](data_policy.json) and [full duplicate lists](dataset.json). All test images remain included.",
             f"Conflicting-label duplicate groups: {len(result['conflicting_label_duplicate_groups'])}. Labels are not automatically corrected.",
             f"Training largest/smallest class ratio: {result['train_imbalance_ratio']:.2f}.", '',
             '![Representative training cells](representative_cells.png)', '![Class distribution](class_distribution.png)', '',
             '## Training decisions', '',
             'EfficientNet-B0: ImageNet transfer learning, eight outputs, 224px input. Its roughly 5.3M original parameters and low operation count suit later Mac inference. Upsampling does not add information.',
             'Evaluation: bicubic resize to 256 then center crop 224, RGB, ImageNet mean/std. Training adds random crop, flips, rotation and mild color jitter. No augmentation of validation/test.',
             'Unweighted cross-entropy preserves natural class frequencies; validation macro F1 gives each class equal weight during checkpoint selection. Per-class recall must be inspected for minority classes.', '',
             '## Limitations', ''] + ['- ' + s for s in result['limitations']]
    (out / 'dataset_report.md').write_text('\n'.join(rows) + '\n')
    print(json.dumps({k: result[k] for k in ['total_images', 'split_totals', 'dimensions', 'ready_for_training', 'train_imbalance_ratio']}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    audit(parser.parse_args().data)
