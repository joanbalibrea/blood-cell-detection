"""Validate YOLO labels, dimensions, class balance and exact split leakage."""
import argparse
import hashlib
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ['WBC', 'RBC', 'Platelets']
ROOT = Path(__file__).resolve().parent

def audit(data):
    report = {'classes': CLASSES, 'splits': {}, 'errors': [], 'cross_split_duplicates': []}
    hashes = defaultdict(list)
    for split in ('train', 'val', 'test'):
        counts, sizes, modes = Counter(), Counter(), Counter()
        images = sorted((data / 'images' / split).glob('*.png'))
        labels = {p.stem for p in (data / 'labels' / split).glob('*.txt')}
        for image in images:
            raw = image.read_bytes()
            width, height = struct.unpack('>II', raw[16:24])
            sizes[f'{width}x{height}'] += 1
            modes[str(raw[25])] += 1
            hashes[hashlib.sha256(raw).hexdigest()].append(f'{split}/{image.name}')
            label = data / 'labels' / split / (image.stem + '.txt')
            if not label.exists():
                report['errors'].append(f'Missing label: {image.name}')
                continue
            for line_number, line in enumerate(label.read_text().splitlines(), 1):
                try:
                    c, x, y, w, h = map(float, line.split())
                    assert c.is_integer() and 0 <= c < 3
                    assert all(0 <= v <= 1 for v in (x, y, w, h)) and w > 0 and h > 0
                    assert x-w/2 >= -1e-5 and y-h/2 >= -1e-5
                    assert x+w/2 <= 1.00001 and y+h/2 <= 1.00001
                    counts[CLASSES[int(c)]] += 1
                except (ValueError, AssertionError):
                    report['errors'].append(f'{split}/{label.name}:{line_number}: invalid box')
        orphan = labels - {p.stem for p in images}
        report['errors'].extend(f'{split}/{name}: orphan label' for name in sorted(orphan))
        report['splits'][split] = {'images': len(images), 'objects': sum(counts.values()),
                                  'classes': dict(counts), 'dimensions': dict(sizes),
                                  'png_color_types': dict(modes)}
    report['cross_split_duplicates'] = [v for v in hashes.values() if len({p.split('/')[0] for p in v}) > 1]
    report['total_images'] = sum(s['images'] for s in report['splits'].values())
    report['total_objects'] = sum(s['objects'] for s in report['splits'].values())
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT.parent / 'TXL-PBC')
    args = parser.parse_args()
    report = audit(args.data)
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'dashboard/data').mkdir(parents=True, exist_ok=True)
    for output in ('reports/dataset.json', 'dashboard/data/dataset.json'):
        (ROOT / output).write_text(json.dumps(report, indent=2))
    lines = ['# Dataset audit', '', '| Split | Images | WBC | RBC | Platelets |', '|---|---:|---:|---:|---:|']
    for name, s in report['splits'].items():
        lines.append(f"| {name} | {s['images']} | " + ' | '.join(str(s['classes'].get(c, 0)) for c in CLASSES) + ' |')
    lines += ['', f"Total: {report['total_images']} images, {report['total_objects']} objects.",
              f"Invalid/missing labels: {len(report['errors'])}.",
              f"Exact duplicates across splits: {len(report['cross_split_duplicates'])}.",
              '', 'No patient identifiers are available, so patient-level independence cannot be verified.',
              'The three labels identify broad cell types; WBC subtype classification is not supported.',
              'PNG color types: 2 = RGB, 6 = RGBA. Training inputs are converted to RGB without changing geometry.',
              'Original splits are preserved; the test split is reserved for final evaluation.']
    (ROOT / 'reports/dataset.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: report[k] for k in ('total_images', 'total_objects', 'errors', 'cross_split_duplicates')}))
