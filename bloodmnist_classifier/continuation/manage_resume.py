"""Submit and monitor exactly one continuation using existing private Kaggle inputs."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from training.common import save_json


def now():
    return datetime.now(timezone.utc).isoformat()


def state():
    return json.loads((HERE / 'run.json').read_text()) if (HERE / 'run.json').exists() else {}


def api():
    from dotenv import load_dotenv
    load_dotenv(ROOT.parent / '.env', override=False)
    from kaggle.api.kaggle_api_extended import KaggleApi
    client = KaggleApi(); client.authenticate()
    return client


def publish(**updates):
    file = ROOT / 'dashboard/data/status.json'
    value = json.loads(file.read_text()) if file.exists() else {}
    value.update(updates); save_json(file, value)


def prepare():
    if state().get('submission_attempted_at'):
        raise RuntimeError('Continuation submission already attempted; inspect status instead of preparing another job')
    original = json.loads((ROOT / 'run.json').read_text())
    previous = ROOT / 'outputs/download/v2/cellscope'
    required = ['last.pt', 'best.pt', 'history.json', 'evaluation.json']
    if not all((previous / name).is_file() for name in required):
        raise RuntimeError('Download original completed checkpoints first')
    history = json.loads((previous / 'history.json').read_text())
    if history[-1]['epoch'] != 33:
        raise RuntimeError('This reviewed continuation expects the observed epoch-33 checkpoint')
    sources = []
    for filename in ['common.py', 'evaluation.py']:
        source = (ROOT / 'training' / filename).read_text()
        sources.append(''.join(line for line in source.splitlines(keepends=True) if not line.startswith('from common import ')))
    original_source = (ROOT / 'training/train.py').read_text()
    resolver = next(n for n in ast.parse(original_source).body if isinstance(n, ast.FunctionDef) and n.name == 'resolve_kaggle_inputs')
    sources.append(ast.get_source_segment(original_source, resolver))
    sources.append(''.join(line for line in (HERE / 'resume.py').read_text().splitlines(keepends=True)
                           if not line.startswith(('from common import ', 'from evaluation import '))))
    source = '\n\n'.join(sources)
    compile(source, 'continue_40.py', 'exec')
    (HERE / 'continue_40.py').write_text(source)
    save_json(HERE / 'continue_40.ipynb', {'nbformat': 4, 'nbformat_minor': 5,
              'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
              'cells': [{'id': 'continuation-info', 'cell_type': 'markdown', 'metadata': {}, 'source': [
                  '# CellScope: continue epochs 34–40\n',
                  'Restores model/optimizer/scheduler/scaler from the original private version-2 output. Early stopping disabled only for this seven-epoch continuation. No dataset or checkpoint re-upload.\n']},
                  {'id': 'continuation-code', 'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source.splitlines(keepends=True)}]})
    user = original['kernel'].split('/')[0]
    source_ref = original['kernel'] + '/2'
    kernel = user + '/cellscope-bloodmnist-continue-40'
    save_json(HERE / 'kernel-metadata.json', {'id': kernel, 'title': 'CellScope BloodMNIST Continue 40',
              'code_file': 'continue_40.ipynb', 'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
              'enable_gpu': True, 'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4',
              'dataset_sources': [original['dataset']], 'kernel_sources': [source_ref], 'competition_sources': []})
    save_json(HERE / 'run.json', {'kernel': kernel, 'source_kernel': source_ref, 'source_completed_epoch': 33,
              'target_epoch': 40, 'prepared_at': now(), 'dataset': original['dataset']})
    save_json(HERE / 'live_history.json', history)
    print('Prepared continuation using existing private dataset and notebook output; no local payload upload required.')


def ref():
    run = state()
    if not run.get('version'):
        raise RuntimeError('No confirmed continuation version; inspect Kaggle before any retry')
    return run['kernel'] + '/' + str(run['version'])


def launch(client):
    run = state()
    if run.get('submission_attempted_at'):
        raise RuntimeError('Continuation already attempted. Duplicate continuation jobs are disabled.')
    status = str(client.kernels_status(run['source_kernel']).status).lower()
    if 'complete' not in status:
        raise RuntimeError('Original source job is not complete; do not launch a concurrent continuation')
    with (HERE / 'submission.lock').open('x') as file:
        file.write(now())
    run['submission_attempted_at'] = now(); save_json(HERE / 'run.json', run)
    response = client.kernels_push(str(HERE), timeout='3600', acc='NvidiaTeslaT4')
    if getattr(response, 'error', None) or getattr(response, 'version_number', None) is None:
        raise RuntimeError('No confirmed successful continuation submission. Inspect Kaggle; do not blindly retry.')
    run.update({'version': int(response.version_number), 'submitted_at': now()}); save_json(HERE / 'run.json', run)
    publish(status='pending', version=run['version'], kernel_url='https://www.kaggle.com/code/' + run['kernel'],
            note='Continuing from epoch 33 to 40. Existing test results belong to the previously selected checkpoint until the continuation completes.')
    print('Continuation submitted:', 'https://www.kaggle.com/code/' + run['kernel'], 'version', run['version'])


def copy_atomic(src, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.download-tmp')
    shutil.copy2(src, temp); temp.replace(destination)


def sync(client):
    run = state(); status = str(client.kernels_status(ref()).status).rsplit('.', 1)[-1].lower()
    terminal = any(s in status for s in ['complete', 'error', 'fail', 'cancel'])
    if run.get('downloaded_at'):
        return terminal
    publish(status='running' if 'running' in status else ('pending' if not terminal else 'pending'), kaggle_status=status,
            synced_at=now(), kernel_url='https://www.kaggle.com/code/' + run['kernel'],
            note='Continuation to 40 epochs. Curves use observed logs; complete saved metrics synchronize after Kaggle saves outputs. Existing test metrics refer to the earlier checkpoint until synchronization finishes.')
    if not terminal:
        print('Continuation status:', status, flush=True); return False
    destination = ROOT / 'outputs/download' / ('continuation-v' + str(run['version']))
    destination.mkdir(parents=True, exist_ok=True)
    token = None
    while True:
        _, token = client.kernels_output(ref(), str(destination), force=True, page_token=token, page_size=100)
        if not token:
            break
    candidates = [p.parent for p in destination.rglob('progress.json') if p.parent.name == 'cellscope_resume']
    if len(candidates) != 1:
        publish(status='error', note='Continuation ended without its progress artifact. Inspect downloaded logs.'); return True
    output = candidates[0]
    progress = json.loads((output / 'progress.json').read_text())
    if progress.get('status') != 'completed' or progress.get('epoch') != 40:
        publish(status='error', note=progress.get('error', 'Continuation ended before epoch 40. Inspect its diagnostics.'))
        return True
    history = json.loads((output / 'history.json').read_text())
    if [r['epoch'] for r in history] != list(range(1, 41)):
        raise RuntimeError('Downloaded history is not a complete 1–40 epoch sequence')
    for file in output.iterdir():
        if file.is_file() and file.suffix in {'.json', '.csv', '.png', '.md'}:
            copy_atomic(file, ROOT / 'outputs' / file.name)
    if (output / 'examples').exists():
        shutil.copytree(output / 'examples', ROOT / 'outputs/examples', dirs_exist_ok=True)
    for name in ['best.pt', 'model_config.json', 'class_mapping.json']:
        copy_atomic(output / name, ROOT / 'models' / name)
    for name in ['history.json', 'progress.json', 'evaluation.json', 'model_config.json', 'startup.json']:
        copy_atomic(output / name, ROOT / 'dashboard/data' / name)
    import torch
    from inference import CellClassifier
    torch.set_num_threads(2)
    source = Path(json.loads((ROOT / 'reports/generated/dataset.json').read_text())['source'])
    sample = next(iter(sorted((source / 'train/basophil').glob('*.png'))))
    prediction = CellClassifier(ROOT / 'models/best.pt', device='cpu').predict(sample)
    save_json(HERE / 'local_inference.json', {'status': 'passed', 'device': 'cpu', 'prediction': prediction,
              'note': 'Functional CPU inference on actual downloaded selected weights; not another test-set evaluation.'})
    run['downloaded_at'] = now(); save_json(HERE / 'run.json', run)
    result = json.loads((output / 'evaluation.json').read_text())
    publish(status='completed', synced_at=now(), note='All 40 epochs verified and synchronized. ' + result['continuation_evaluation_note'])
    print('Verified 40/40 epochs; all metrics synchronized; actual local CPU inference passed.', flush=True)
    return True


def stream_once(client):
    history = {r['epoch']: r for r in json.loads((HERE / 'live_history.json').read_text())}
    buffer = ''
    for event in client.kernels_logs_stream(ref()):
        if state().get('downloaded_at'):
            return
        buffer += event.get('data', '')
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            if 'CELLSCOPE_STARTUP ' in line:
                try:
                    save_json(ROOT / 'dashboard/data/startup.json', json.loads(line.split('CELLSCOPE_STARTUP ', 1)[1]))
                    print('Observed continuation CUDA diagnostics.', flush=True)
                except json.JSONDecodeError:
                    pass
            if 'CELLSCOPE_METRIC ' not in line:
                continue
            try:
                row = json.loads(line.split('CELLSCOPE_METRIC ', 1)[1])
            except json.JSONDecodeError:
                continue
            if not 34 <= row.get('epoch', 0) <= 40:
                continue
            history[row['epoch']] = row; rows = [history[k] for k in sorted(history)]
            save_json(HERE / 'live_history.json', rows); save_json(ROOT / 'dashboard/data/history.json', rows)
            best = max(rows, key=lambda r: r['val_macro_f1'])['epoch']
            save_json(ROOT / 'dashboard/data/progress.json', {'status': 'running', 'best_epoch': best, **row})
            publish(status='running', metrics_observed_at=now(), note='Measured continuation epochs received from Kaggle logs; final outputs will synchronize automatically.')
            print('Observed continuation epoch', row['epoch'], flush=True)


def logs(client):
    from requests.exceptions import RequestException
    for attempt in range(3):
        if state().get('downloaded_at'):
            return
        try:
            stream_once(client)
            if any(x in str(client.kernels_status(ref()).status).lower() for x in ['complete', 'error', 'fail', 'cancel']):
                return
        except RequestException as exc:
            print('Log connection interrupted:', type(exc).__name__, flush=True)
        time.sleep(5)
    print('Live stream unavailable; status/output watcher remains the reliable fallback.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'launch', 'sync', 'watch', 'logs'])
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare()
        else:
            client = api()
            if args.command == 'launch':
                launch(client)
            elif args.command == 'sync':
                sync(client)
            elif args.command == 'logs':
                logs(client)
            else:
                while not sync(client):
                    time.sleep(30)
    except KeyboardInterrupt:
        print('Local monitor stopped; remote training is unaffected.')
    except RuntimeError as exc:
        print(str(exc)); raise SystemExit(1)
    except Exception as exc:
        print('Operation failed:', type(exc).__name__, '(external details withheld to protect credentials). No automatic resubmission.'); raise SystemExit(1)
