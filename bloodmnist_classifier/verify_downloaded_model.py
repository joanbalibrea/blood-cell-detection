"""Wait for the existing sync watcher, then smoke-test real eight-class CPU inference.

Reads checkpoints/status only; never submits, stops, retrains, or changes the
Kaggle job. Verification artifacts have a separate, newly created location.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'reports/generated/local_verification'


def write(value):
    REPORT.mkdir(parents=True, exist_ok=True)
    temp = REPORT / 'verification.tmp'
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(REPORT / 'verification.json')


def verify():
    started = time.monotonic()
    write({'status': 'waiting', 'note': 'Waiting for the existing Kaggle sync watcher to download a real eight-class best.pt. No inference has run yet.'})
    while time.monotonic() - started < 22500:
        checkpoint = ROOT / 'models/best.pt'
        evaluation = ROOT / 'outputs/evaluation.json'
        status_file = ROOT / 'dashboard/data/status.json'
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        run_file = ROOT / 'run.json'
        run = json.loads(run_file.read_text()) if run_file.exists() else {}
        # sync publishes downloaded evaluation only after checkpoint copying finishes.
        if checkpoint.exists() and evaluation.exists() and status.get('status') == 'completed' and run.get('outputs_downloaded_at'):
            import torch
            from inference import CellClassifier
            torch.set_num_threads(2)
            audit = json.loads((ROOT / 'reports/generated/dataset.json').read_text())
            source = Path(audit['source'])
            image = next(iter(sorted((source / 'train/basophil').glob('*.png'))))
            classifier = CellClassifier(checkpoint, device='cpu')
            if classifier.config.get('manifest_sha256') != audit['manifest_sha256']:
                raise ValueError('Downloaded checkpoint manifest does not match the audited eight-class experiment')
            tick = time.perf_counter()
            result = classifier.predict(image)
            elapsed = time.perf_counter() - tick
            if len(result['probabilities']) != 8 or not all(math.isfinite(p) for p in result['probabilities'].values()) or abs(sum(result['probabilities'].values()) - 1) > 1e-5:
                raise ValueError('Invalid eight-class inference probabilities')
            write({'status': 'passed', 'checked_at': datetime.now(timezone.utc).isoformat(),
                   'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   'device': 'cpu', 'image': image.relative_to(source).as_posix(),
                   'prediction_seconds': elapsed, 'prediction': result,
                   'note': 'Actual downloaded checkpoint loaded strictly and produced eight finite normalized probabilities on a training example. This is a functional inference check, not a new performance estimate or another test-set evaluation.'})
            print('Actual downloaded eight-class checkpoint passed local CPU inference verification.', flush=True)
            return
        if status.get('status') == 'error':
            write({'status': 'pending_remote_error', 'note': 'No local trained-weight check was possible; the remote run reported an error. Nothing was restarted or cancelled.'})
            return
        time.sleep(30)
    write({'status': 'waiting_timeout', 'note': 'No finalized checkpoint was available during the waiting window. No remote job was stopped or restarted.'})


if __name__ == '__main__':
    try:
        verify()
    except Exception as exc:
        write({'status': 'error', 'error_type': type(exc).__name__,
               'note': 'Local verification failed; training and checkpoint files were not changed.'})
        raise
