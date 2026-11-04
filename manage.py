"""Private Kaggle dataset submission, GPU job launch and dashboard synchronization."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from PIL import Image
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)

def api():
    load_dotenv(ROOT / '.env')
    from kaggle.api.kaggle_api_extended import KaggleApi
    client = KaggleApi()
    client.authenticate()
    return client

def prepare(client):
    user = client.get_config_value('username')
    ref = f'{user}/txl-pbc-private-detection'
    kernel = f'{user}/txl-pbc-yolo26-training'
    folder = ROOT / 'staging/dataset'
    folder.mkdir(parents=True, exist_ok=True)
    source = ROOT.parent / 'TXL-PBC'
    for split in ('train', 'val', 'test'):
        dest = folder / 'images' / split
        dest.mkdir(parents=True, exist_ok=True)
        for path in sorted((source / 'images' / split).glob('*.png')):
            with Image.open(path) as im:
                im.convert('RGB').save(dest / path.name)
        shutil.copytree(source / 'labels' / split, folder / 'labels' / split, dirs_exist_ok=True)
    (folder / 'data.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\nnames: [WBC, RBC, Platelets]\n')
    save(folder / 'dataset-metadata.json', {'id': ref, 'title': 'TXL PBC Private Detection',
         'licenses': [{'name': 'other'}], 'description': 'Private research copy. Original rights retained; no new reuse license granted.'})
    save(ROOT / 'kaggle/kernel-metadata.json', {'id': kernel, 'title': 'TXL PBC YOLO26 Training',
         'code_file': 'train.py', 'language': 'python', 'kernel_type': 'script',
         'is_private': True, 'enable_gpu': True, 'enable_internet': True,
         'machine_shape': 'NvidiaTeslaT4',
         'docker_image': 'gcr.io/kaggle-gpu-images/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01',
         'docker_image_pinning_type': 'original',
         'dataset_sources': [ref], 'competition_sources': [], 'kernel_sources': []})
    save(ROOT / 'run.json', {'dataset': ref, 'kernel': kernel})
    print('Prepared private dataset and kernel:', ref, kernel)

def sync(client):
    run = json.loads((ROOT / 'run.json').read_text())
    ref = run['kernel'] + (f"/{run['version']}" if run.get('version') else '')
    response = client.kernels_status(ref)
    status = str(response.status)
    state = {'status': status.rsplit('.', 1)[-1].lower(), 'synced_at': datetime.now(timezone.utc).isoformat(),
             'kernel_url': f"https://www.kaggle.com/code/{run['kernel']}",
             'failure_message': getattr(response, 'failure_message', None),
             'note': 'Live epoch callbacks update charts when the log watcher is active. Final results synchronize after Kaggle saves outputs.'}
    save(ROOT / 'dashboard/data/status.json', state)
    if 'complete' in status.lower() or 'error' in status.lower():
        token = None
        while True:
            _, token = client.kernels_output(ref, str(ROOT / 'outputs'), force=True,
                                              page_token=token, page_size=100)
            if not token:
                break
        for filename in ('progress.json', 'performance.json'):
            p = ROOT / 'outputs' / filename
            if p.exists():
                save(ROOT / 'dashboard/data' / filename, json.loads(p.read_text()))
        performance = ROOT / 'outputs/performance.json'
        if performance.exists():
            report = json.loads(performance.read_text())
            save(ROOT / 'reports/performance.json', report)
            rows = ['# Final test performance', '', 'Evaluated once on the held-out test split using validation-selected best weights.', '',
                    '| Class | Precision | Recall | mAP50 | mAP50–95 |', '|---|---:|---:|---:|---:|']
            for item in report['per_class']:
                rows.append('| ' + item['name'] + ' | ' + ' | '.join(f"{item[k]:.4f}" for k in ('precision', 'recall', 'mAP50', 'mAP50_95')) + ' |')
            rows += ['', '## Aggregate metrics', '']
            rows += [f'- {key}: {float(value):.4f}' for key, value in report['metrics'].items()]
            rows += ['', '## Runtime', '',
                     f"- Completed epochs: {report['training']['epoch']}",
                     f"- GPU: {report['training']['gpu']}",
                     f"- End-to-end elapsed minutes: {report['elapsed_seconds']/60:.2f}",
                     f"- Ultralytics: {report['training']['ultralytics']}",
                     f"- PyTorch: {report['training']['torch']}"]
            rows += [f'- {key}: {float(value):.3f} ms/image' for key, value in report['speed_ms'].items()]
            rows += ['', 'Inference timing reflects this GPU and evaluation batch configuration.',
                     'Original splits were preserved; patient-level independence could not be verified.',
                     'The 49 platelet test annotations make minority-class estimates sensitive to individual detections.',
                     '', 'See downloaded `evaluation/test/` for confusion matrices and precision–recall curves.']
            (ROOT / 'reports/performance.md').write_text('\n'.join(rows) + '\n')
        csv = ROOT / 'outputs/runs/blood_cells/results.csv'
        if csv.exists():
            shutil.copy2(csv, ROOT / 'dashboard/data/results.csv')
            shutil.copy2(csv, ROOT / 'reports/training_history.csv')
        log = ROOT / 'outputs/txl-pbc-yolo26-training.log'
        if log.exists():
            state['log_available'] = True
        startup = ROOT / 'outputs/startup.json'
        if startup.exists():
            diagnostics = json.loads(startup.read_text())
            save(ROOT / 'reports/startup.json', diagnostics)
            if not diagnostics.get('cuda_available'):
                state['failure_message'] = 'Kaggle session started without a usable CUDA GPU; training did not run.'
                state['note'] = state['failure_message']
    save(ROOT / 'dashboard/data/status.json', state)
    print(json.dumps(state))
    return status

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'upload', 'launch', 'sync', 'watch'])
    args = parser.parse_args()
    client = api()
    if args.command == 'prepare':
        prepare(client)
    elif args.command == 'upload':
        result = client.dataset_create_new(str(ROOT / 'staging/dataset'), public=False, quiet=True,
                                          convert_to_csv=False, dir_mode='zip')
        print(result)
    elif args.command == 'launch':
        response = client.kernels_push(str(ROOT / 'kaggle'), timeout='21600', acc='NvidiaTeslaT4')
        print(response)
        if getattr(response, 'error', None):
            raise RuntimeError('Kaggle rejected the kernel submission')
        run = json.loads((ROOT / 'run.json').read_text())
        run['version'] = int(response.version_number)
        save(ROOT / 'run.json', run)
        save(ROOT / 'dashboard/data/status.json', {'status': 'submitted',
             'synced_at': datetime.now(timezone.utc).isoformat(),
             'kernel_url': f"https://www.kaggle.com/code/{run['kernel']}",
             'note': 'Waiting for the newly submitted GPU session to start.'})
    elif args.command == 'sync':
        sync(client)
    else:
        while True:
            status = sync(client)
            if any(s in status.lower() for s in ('complete', 'error', 'cancel')):
                break
            time.sleep(45)
