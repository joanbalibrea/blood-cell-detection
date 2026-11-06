"""Private Kaggle submission, conservative synchronization and local-only serving.

No credentials are copied to this subproject or printed. External errors are
reported by type only because arbitrary HTTP exception text may contain secrets.
"""
import argparse
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
from urllib.parse import unquote, urlsplit

from training.common import save_json

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def api():
    from dotenv import load_dotenv
    load_dotenv(ROOT.parent / '.env', override=False)
    from kaggle.api.kaggle_api_extended import KaggleApi
    client = KaggleApi()
    client.authenticate()
    return client


def state():
    return json.loads(RUN.read_text()) if RUN.exists() else {}


def publish_status(**updates):
    p = ROOT / 'dashboard/data/status.json'
    value = json.loads(p.read_text()) if p.exists() else {}
    value.update(updates)
    save_json(p, value)


def prepare(client, overlap_policy='exclude'):
    audit_file = ROOT / 'reports/generated/dataset.json'
    if not audit_file.exists():
        raise RuntimeError('Run audit_dataset.py first')
    audit = json.loads(audit_file.read_text())
    if not audit['integrity_passed']:
        raise RuntimeError('Dataset integrity checks failed; inspect the audit')
    if state().get('submission_attempted_at'):
        raise RuntimeError('A submission already exists; do not change the prepared assets for this run.')
    source = Path(audit['source'])
    assets = ROOT / 'staging/assets'; assets.mkdir(parents=True, exist_ok=True)
    policy = audit['recommended_exclusions'] if overlap_policy == 'exclude' else {
        'name': 'original_all_images_with_known_leakage', 'excluded_paths': [],
        'keep_test_intact': True, 'source_files_modified': False,
        'reason': 'All original images retained; cross-split duplicates contaminate validation/test estimates.'}
    save_json(assets / 'data_policy.json', policy)
    save_json(ROOT / 'reports/generated/selected_data_policy.json', policy)
    for filename, src in [('manifest.csv', ROOT / 'reports/generated/manifest.csv'),
                          ('config.json', ROOT / 'training/config.json')]:
        shutil.copy2(src, assets / filename)
    archive = assets / 'bloodmnist.zip'
    import csv
    import hashlib
    with (assets / 'manifest.csv').open() as file:
        manifest = list(csv.DictReader(file))
    if not archive.exists():
        temp = archive.with_suffix('.tmp')
        with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for row in manifest:
                data = (source / row['path']).read_bytes()
                if hashlib.sha256(data).hexdigest() != row['sha256']:
                    raise RuntimeError('Source changed since audit: ' + row['path'])
                z.writestr(row['path'], data)
        temp.replace(archive)
    else:
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist()) != {r['path'] for r in manifest}:
                raise RuntimeError('Existing transport archive does not match manifest')
            for row in manifest:
                if hashlib.sha256(z.read(row['path'])).hexdigest() != row['sha256']:
                    raise RuntimeError('Transport archive checksum mismatch')
    from torchvision.models import EfficientNet_B0_Weights
    # One necessary offline copy, placed only in ignored staging, never in original data.
    EfficientNet_B0_Weights.IMAGENET1K_V1.get_state_dict(progress=False, check_hash=True, model_dir=str(assets))
    weight_path = assets / 'efficientnet_b0_rwightman-7f5810bc.pth'
    if not hashlib.sha256(weight_path.read_bytes()).hexdigest().startswith('7f5810bc'):
        raise RuntimeError('Official pretrained weights checksum mismatch')
    user = client.get_config_value('username')
    dataset = f'{user}/cellscope-bloodmnist-classifier-assets'
    kernel = f'{user}/cellscope-bloodmnist-classifier'
    save_json(assets / 'dataset-metadata.json', {'id': dataset, 'title': 'CellScope BloodMNIST Classifier Assets',
              'licenses': [{'name': 'other'}],
              'description': 'Private research transport copy of supplied BloodMNIST images and ImageNet pretrained weights. Original rights retained; no new reuse rights granted.'})
    save_json(ROOT / 'reports/generated/offline_assets.json', {
        'destination': dataset, 'private': True, 'contains_credentials': False,
        'files': [{'name': p.name, 'bytes': p.stat().st_size,
                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(assets.iterdir()) if p.is_file()],
        'note': 'One compressed transport copy; original image files remain in place. Manifest contains relative paths, labels and hashes.'})
    bundle_sources()
    save_json(ROOT / 'training/kernel-metadata.json', {'id': kernel, 'title': 'CellScope BloodMNIST Classifier',
              'code_file': 'bloodmnist_training.ipynb', 'language': 'python', 'kernel_type': 'notebook',
              'is_private': True, 'enable_gpu': True, 'enable_internet': False,
              'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': [dataset],
              'competition_sources': [], 'kernel_sources': []})
    previous = state()
    save_json(RUN, {**previous, 'dataset': dataset, 'kernel': kernel, 'prepared_at': now(),
                    'overlap_policy': policy['name'], 'manifest_sha256': audit['manifest_sha256']})
    initial_config = json.loads((assets / 'config.json').read_text())
    initial_config['data_policy'] = policy
    initial_config['effective_split_sizes'] = {s: sum(1 for r in manifest if r['split'] == s and r['path'] not in set(policy['excluded_paths'])) for s in ('train', 'val')}
    from training.common import build_model
    initial_config['parameter_count'] = sum(p.numel() for p in build_model().parameters())
    save_json(ROOT / 'dashboard/data/model_config.json', initial_config)
    for name, value in [('history.json', []), ('progress.json', None), ('evaluation.json', None), ('startup.json', None)]:
        path = ROOT / 'dashboard/data' / name
        if not path.exists():
            save_json(path, value)
    publish_status(status='pending', note='Prepared; no GPU job submitted yet. Epoch metrics are not live.',
                   planned_epochs=40)
    print('Prepared independently runnable offline notebook; private dataset:', dataset)


def bundle_sources():
    chunks = []
    for name in ['common.py', 'evaluation.py', 'train.py']:
        lines = (ROOT / 'training' / name).read_text().splitlines(keepends=True)
        chunks.append(''.join(line for line in lines if not line.startswith(('from common import ', 'from evaluation import '))))
    source = '\n\n'.join(chunks)
    compile(source, 'kaggle_train.py', 'exec')
    (ROOT / 'training/kaggle_train.py').write_text(source)
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'cells': [{'id': 'overview', 'cell_type': 'markdown', 'metadata': {},
                           'source': ['# CellScope — eight-class BloodMNIST classifier\n',
                                      'Offline ImageNet transfer learning. CUDA required. Original test split evaluated once after validation model selection.\n']},
                          {'id': 'pipeline', 'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                           'outputs': [], 'source': source.splitlines(keepends=True)}]}
    save_json(ROOT / 'training/bloodmnist_training.ipynb', notebook)


def upload(client):
    run = state()
    if not run.get('prepared_at'):
        raise RuntimeError('Prepare before upload')
    if run.get('upload_attempted_at'):
        print('Upload previously attempted. Checking remote processing state instead of uploading again.')
        print('Dataset status:', client.dataset_status(run['dataset']))
        return
    run['upload_attempted_at'] = now(); save_json(RUN, run)
    response = client.dataset_create_new(str(ROOT / 'staging/assets'), public=False, quiet=True, convert_to_csv=False, dir_mode='skip')
    if getattr(response, 'error', None):
        raise RuntimeError('Kaggle dataset upload rejected; inspect the private dataset page.')
    run['uploaded_at'] = now(); save_json(RUN, run)
    print('Private dataset upload accepted. Processing status:', client.dataset_status(run['dataset']))


def launch(client, retry_failed=False):
    run = state()
    if run.get('submission_attempted_at'):
        if not retry_failed:
            raise RuntimeError('Submission already attempted. Use sync and inspect Kaggle; duplicate jobs are disabled.')
        if not run.get('version') or not run.get('outputs_downloaded_at'):
            raise RuntimeError('Download and inspect failed-run diagnostics before a corrected retry.')
        previous_status = str(client.kernels_status(ref_for(run)).status).lower()
        if not any(s in previous_status for s in ('error', 'fail', 'cancel')):
            raise RuntimeError('Only a confirmed failed/stopped version may be retried. Running/completed jobs cannot be duplicated.')
        old_notebook = ROOT / 'training/bloodmnist_training.ipynb'
        import hashlib
        previous_hash = hashlib.sha256(old_notebook.read_bytes()).hexdigest()
        bundle_sources()
        if hashlib.sha256(old_notebook.read_bytes()).hexdigest() == previous_hash:
            raise RuntimeError('No source fix detected since the failed notebook. Retry refused.')
    if not run.get('uploaded_at'):
        raise RuntimeError('Upload assets before launching')
    remote = str(client.dataset_status(run['dataset'])).lower()
    if not any(k in remote for k in ('ready', 'complete')):
        raise RuntimeError('Private dataset is still processing; check status before launching.')
    # Record before network call so an ambiguous timeout cannot create an accidental second run.
    lock = ROOT / 'staging' / (f"submission-retry-after-v{run['version']}.lock" if retry_failed else 'submission.lock')
    with lock.open('x') as f:
        f.write(now())
    if retry_failed:
        save_json(ROOT / 'outputs' / f"run_v{run['version']}.json", run)
        run.pop('outputs_downloaded_at', None)
    run['submission_attempted_at'] = now(); save_json(RUN, run)
    publish_status(status='pending', note='Submission requested. Waiting for confirmed CUDA startup; no live metrics yet.')
    result = client.kernels_push(str(ROOT / 'training'), timeout='21600', acc='NvidiaTeslaT4')
    if getattr(result, 'error', None):
        publish_status(status='error', note='Kaggle rejected submission. Inspect account GPU permissions/quota in Kaggle.')
        raise RuntimeError('Kaggle rejected the GPU submission. No retry was submitted.')
    version = getattr(result, 'version_number', None)
    if version is None:
        raise RuntimeError('Kaggle response had no version; inspect Kaggle before any retry.')
    run.update({'version': int(version), 'submitted_at': now()}); save_json(RUN, run)
    publish_status(status='pending', version=run['version'], kernel_url='https://www.kaggle.com/code/' + run['kernel'],
                   note='GPU job submitted. Status-only monitoring until actual epoch log records or saved outputs become available.')
    print('Submitted one private GPU job:', 'https://www.kaggle.com/code/' + run['kernel'], 'version', version)


def ref_for(run):
    if not run.get('version'):
        raise RuntimeError('No confirmed submitted version. Inspect the Kaggle notebook before any retry.')
    return run['kernel'] + '/' + str(run['version'])


def adopt(client, version):
    if version is None or version < 1:
        raise RuntimeError('Supply --version with the existing Kaggle notebook version number.')
    run = state()
    if not run.get('kernel'):
        raise RuntimeError('Prepare the project first.')
    client.kernels_status(run['kernel'] + '/' + str(version))
    run.update({'version': version, 'adopted_at': now(), 'submission_attempted_at': run.get('submission_attempted_at', now())})
    run.pop('outputs_downloaded_at', None)
    save_json(RUN, run)
    publish_status(status='pending', kernel_url='https://www.kaggle.com/code/' + run['kernel'],
                   note='Monitoring an existing Kaggle version. No job was launched by adopt.')
    print('Now monitoring existing version', version)


def sync(client):
    run = state(); ref = ref_for(run)
    response = client.kernels_status(ref)
    raw = str(response.status).rsplit('.', 1)[-1].lower()
    terminal = any(x in raw for x in ('complete', 'error', 'cancel', 'fail'))
    status = 'running' if 'running' in raw else 'pending'
    if terminal:
        status = 'completed' if 'complete' in raw else 'error'
    publish_status(status=status, kaggle_status=raw, synced_at=now(),
                   kernel_url='https://www.kaggle.com/code/' + run['kernel'],
                   note='Kaggle status checked. Saved outputs normally become downloadable only after the run ends; epoch curves update only from observed logs or downloaded files.')
    if terminal and not run.get('outputs_downloaded_at'):
        destination = ROOT / 'outputs/download' / ('v' + str(run['version'])); destination.mkdir(parents=True, exist_ok=True)
        token = None
        while True:
            _, token = client.kernels_output(ref, str(destination), force=True, page_token=token, page_size=100)
            if not token:
                break
        progress_files = list(destination.rglob('progress.json'))
        if progress_files:
            remote_out = progress_files[0].parent
            for filename in ['history.json', 'history.csv', 'progress.json', 'evaluation.json', 'model_config.json', 'startup.json', 'class_mapping.json',
                             'training_curves.png', 'confusion_matrix.png', 'normalized_confusion_matrix.png',
                             'confusion_matrix.csv', 'normalized_confusion_matrix.csv', 'predictions.csv', 'per_class.csv', 'evaluation_report.md']:
                src = remote_out / filename
                if src.exists():
                    shutil.copy2(src, ROOT / 'outputs' / filename)
            if (remote_out / 'examples').exists():
                shutil.copytree(remote_out / 'examples', ROOT / 'outputs/examples', dirs_exist_ok=True)
            for filename in ['best.pt', 'model_config.json', 'class_mapping.json']:
                src = remote_out / filename
                if src.exists():
                    (ROOT / 'models').mkdir(exist_ok=True)
                    shutil.copy2(src, ROOT / 'models' / filename)
            progress = json.loads((remote_out / 'progress.json').read_text())
            if progress.get('status') == 'error':
                publish_status(status='error', note=progress.get('error', 'Remote training failed.'))
            for filename in ['history.json', 'progress.json', 'evaluation.json', 'model_config.json', 'startup.json']:
                src = ROOT / 'outputs' / filename
                if src.exists():
                    shutil.copy2(src, ROOT / 'dashboard/data' / filename)
        # A completed notebook without verified evaluation is never reported as a successful model.
        verified_result = bool(progress_files and (progress_files[0].parent / 'evaluation.json').exists() and (progress_files[0].parent / 'best.pt').exists())
        if status == 'completed' and not verified_result:
            publish_status(status='error', note='Kaggle finished, but no verified evaluation artifact was found. Inspect downloaded logs.')
        run['outputs_downloaded_at'] = now(); save_json(RUN, run)
    print('Kaggle:', raw, 'checked at', now())
    return terminal


def stream_logs_once(client):
    run = state(); ref = ref_for(run)
    file = ROOT / 'outputs/live_history.json'
    rows = json.loads(file.read_text()) if file.exists() else []
    history = {r['epoch']: r for r in rows}
    buffer = ''
    for event in client.kernels_logs_stream(ref):
        buffer += event.get('data', '')
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            if 'CELLSCOPE_STARTUP ' in line:
                try:
                    startup = json.loads(line.split('CELLSCOPE_STARTUP ', 1)[1])
                    allowed = ['cuda_available', 'torch_version', 'torchvision_version', 'torch_cuda_build',
                               'gpu_count', 'gpu', 'started_at', 'model_on_cuda', 'batch_on_cuda', 'gpu_execution_verified']
                    save_json(ROOT / 'dashboard/data/startup.json', {k: startup[k] for k in allowed if k in startup})
                    print('Observed CUDA startup diagnostics.', flush=True)
                except json.JSONDecodeError:
                    pass
            if 'CELLSCOPE_METRIC ' not in line:
                continue
            try:
                row = json.loads(line.split('CELLSCOPE_METRIC ', 1)[1])
            except json.JSONDecodeError:
                continue
            keys = ['epoch', 'planned_epochs', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'val_macro_f1', 'learning_rate', 'elapsed_seconds', 'gpu']
            if not all(k in row for k in keys):
                continue
            row = {k: row[k] for k in keys}
            history[row['epoch']] = row
            rows = [history[e] for e in sorted(history)]
            save_json(file, rows); save_json(ROOT / 'dashboard/data/history.json', rows)
            save_json(ROOT / 'dashboard/data/progress.json', {'status': 'running', **row})
            publish_status(status='running', metrics_observed_at=now(),
                           note='Epoch records observed from the Kaggle log stream. Records may be delayed; this is not GPU telemetry.')
            print('Observed epoch', row['epoch'], flush=True)
    print('Log stream ended. Use sync for final saved outputs.')


def logs(client):
    from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout
    for attempt in range(1, 4):
        try:
            stream_logs_once(client)
            status = str(client.kernels_status(ref_for(state())).status).lower()
            if any(s in status for s in ('complete', 'error', 'fail', 'cancel')):
                return
        except (ChunkedEncodingError, ConnectionError, ReadTimeout) as exc:
            print(f'Log connection interrupted ({type(exc).__name__}); reconnect {attempt}/3.', flush=True)
        if attempt < 3:
            time.sleep(5)
    publish_status(note='Epoch log stream is unavailable/interrupted. Use Kaggle console and sync/watch for final saved outputs; displayed epochs are saved records only.')
    print('Log streaming unavailable after bounded reconnects. Status/output synchronization remains available.')


def serve(port):
    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            super().end_headers()

        def send_head(self):
            # Expose only dashboard and reports, never staging, credentials or checkpoint internals.
            target = (ROOT / unquote(urlsplit(self.path).path).lstrip('/')).resolve()
            directories = [ROOT / 'dashboard', ROOT / 'reports/generated', ROOT / 'outputs/examples']
            files = [ROOT / 'outputs' / name for name in ['training_curves.png', 'confusion_matrix.png', 'normalized_confusion_matrix.png']]
            if not (any(target.is_relative_to(d.resolve()) for d in directories) or target in files):
                self.send_error(404); return None
            return super().send_head()
    print(f'CellScope dashboard: http://127.0.0.1:{port}/dashboard/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), partial(Handler, directory=str(ROOT))).serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['auth', 'prepare', 'upload', 'launch', 'retry', 'sync', 'watch', 'logs', 'serve', 'bundle', 'adopt'])
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--overlap-policy', choices=['exclude', 'all'], default='exclude')
    parser.add_argument('--version', type=int)
    args = parser.parse_args()
    try:
        if args.command == 'serve':
            serve(args.port)
        elif args.command == 'bundle':
            bundle_sources(); print('Standalone script and notebook generated.')
        else:
            client = api()
            if args.command in {'sync', 'watch', 'logs'} and (ROOT / 'continuation/run.json').exists():
                from continuation import manage_resume as resumed
                if resumed.state().get('submitted_at'):
                    if args.command == 'sync':
                        resumed.sync(client)
                    elif args.command == 'logs':
                        resumed.logs(client)
                    else:
                        while not resumed.sync(client):
                            time.sleep(30)
                    raise SystemExit(0)
            if args.command == 'auth':
                client.dataset_list(mine=True)
                print('Authenticated read-only Kaggle request succeeded.')
            elif args.command == 'prepare':
                prepare(client, args.overlap_policy)
            elif args.command == 'upload':
                upload(client)
            elif args.command == 'launch':
                launch(client)
            elif args.command == 'retry':
                launch(client, retry_failed=True)
            elif args.command == 'adopt':
                adopt(client, args.version)
            elif args.command == 'sync':
                sync(client)
            elif args.command == 'logs':
                logs(client)
            else:
                while not sync(client):
                    time.sleep(45)
    except KeyboardInterrupt:
        print('Local process stopped. A remote Kaggle job continues independently.')
    except RuntimeError as exc:
        print(str(exc)); raise SystemExit(1)
    except Exception as exc:
        print(f'Operation failed ({type(exc).__name__}). External error text withheld to protect credentials. Check connectivity/account permissions in Kaggle. No automatic resubmission occurred.')
        raise SystemExit(1)
