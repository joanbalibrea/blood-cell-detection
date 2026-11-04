"""Stream Kaggle console events and publish measured PROGRESS callbacks locally."""
import csv
import io
import json
from pathlib import Path
import time
from manage import ROOT, api, save

def publish(state, history):
    save(ROOT / 'dashboard/data/progress.json', state)
    if state.get('phase') == 'training':
        history[int(state['epoch'])] = state
    rows = []
    for epoch, item in sorted(history.items()):
        metrics = item.get('metrics', {})
        rows.append({'epoch': epoch,
                     'train/box_loss': item.get('train_loss', {}).get('train/box_loss', ''),
                     'train/cls_loss': item.get('train_loss', {}).get('train/cls_loss', ''),
                     'metrics/mAP50(B)': metrics.get('metrics/mAP50(B)', ''),
                     'metrics/mAP50-95(B)': metrics.get('metrics/mAP50-95(B)', '')})
    if rows and not (ROOT / 'outputs/runs/blood_cells/results.csv').exists():
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        tmp = ROOT / 'dashboard/data/results.tmp'
        tmp.write_text(buffer.getvalue())
        tmp.replace(ROOT / 'dashboard/data/results.csv')

def main():
    run = json.loads((ROOT / 'run.json').read_text())
    ref = run['kernel'] + (f"/{run['version']}" if run.get('version') else '')
    history, seen = {}, 0
    client = api()
    (ROOT / 'outputs').mkdir(exist_ok=True)
    failures = 0
    while failures < 5:
        count = 0
        try:
            with (ROOT / f"outputs/live-v{run.get('version', 'latest')}.log").open('a') as log:
                for event in client.kernels_logs_stream(ref):
                    count += 1
                    if count <= seen:
                        continue
                    seen = count
                    text = event.get('data', '')
                    log.write(text)
                    log.flush()
                    # Each callback prints one JSON object; ignore non-callback console data.
                    for line in text.splitlines():
                        if 'PROGRESS ' in line:
                            try:
                                state = json.loads(line.split('PROGRESS ', 1)[1])
                            except json.JSONDecodeError:
                                continue
                            publish(state, history)
                            print(json.dumps(state), flush=True)
            status = str(client.kernels_status(ref).status).lower()
            if any(s in status for s in ('complete', 'error', 'cancel')):
                return
            time.sleep(15)
        except Exception as error:
            failures += 1
            print(f'Log stream interrupted ({failures}/5): {type(error).__name__}', flush=True)
            time.sleep(5)
    raise RuntimeError('Log stream unavailable after five attempts; use the Kaggle console and manage.py sync.')

if __name__ == '__main__':
    main()
