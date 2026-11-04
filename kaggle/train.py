"""Self-contained Kaggle GPU entrypoint. No credentials belong in this file."""
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

os.environ['YOLO_AUTOINSTALL'] = 'False'
weights = list(Path('/kaggle/input').rglob('yolo26s.pt'))
assert len(weights) == 1, 'Attach the private offline YOLO26 assets dataset.'
packages = weights[0].parent / 'packages'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps',
                       '--find-links', str(packages), 'ultralytics==8.4.32',
                       'ultralytics-thop==2.2.4', 'polars==1.38.1',
                       'polars-runtime-32==1.38.1', 'opencv-python==4.11.0.86'])
import torch
import yaml
from ultralytics import YOLO

OUT = Path('/kaggle/working')
os.chdir(OUT)
shutil.copy2(weights[0].parent / 'yolo26n.pt', OUT / 'yolo26n.pt')
diagnostics = {'torch': torch.__version__, 'torch_cuda_build': torch.version.cuda,
               'cuda_available': torch.cuda.is_available(), 'gpu_count': torch.cuda.device_count()}
try:
    diagnostics['nvidia_smi'] = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=30).stdout
except FileNotFoundError:
    diagnostics['nvidia_smi'] = 'nvidia-smi executable unavailable'
(OUT / 'startup.json').write_text(json.dumps(diagnostics, indent=2))
print('GPU_DIAGNOSTICS ' + json.dumps(diagnostics), flush=True)
assert torch.cuda.is_available(), 'GPU required: enable GPU and check Kaggle quota/verification.'
data_files = list(Path('/kaggle/input').rglob('data.yaml'))
assert len(data_files) == 1, f'Expected one dataset config, found {data_files}'
dataset = data_files[0].parent
cfg = {'path': str(dataset), 'train': 'images/train', 'val': 'images/val',
       'test': 'images/test', 'names': {0: 'WBC', 1: 'RBC', 2: 'Platelets'}}
(OUT / 'data.yaml').write_text(yaml.safe_dump(cfg))
config = dict(model=str(weights[0]), epochs=100, imgsz=640, batch=16, patience=20,
              seed=42, device=0, workers=2, deterministic=True, amp=True,
              project=str(OUT / 'runs'), name='blood_cells', exist_ok=True,
              plots=True, save=True, save_period=10, close_mosaic=10)
started = time.time()
state = {'phase': 'initializing', 'epoch': 0, 'config': dict(config),
         'gpu': torch.cuda.get_device_name(0),
         'ultralytics': importlib.metadata.version('ultralytics'), 'torch': torch.__version__}
def write_state():
    state['elapsed_seconds'] = time.time() - started
    tmp = OUT / 'progress.tmp'
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(OUT / 'progress.json')
    print('PROGRESS ' + json.dumps(state, default=str), flush=True)

def on_epoch(trainer):
    state.update(phase='training', epoch=int(trainer.epoch) + 1,
                 metrics={k: float(v) for k, v in trainer.metrics.items()},
                 train_loss={k: float(v) for k, v in trainer.label_loss_items(trainer.tloss, prefix='train').items()})
    write_state()

write_state()
try:
    model = YOLO(config.pop('model'))
    model.add_callback('on_fit_epoch_end', on_epoch)
    model.train(data=str(OUT / 'data.yaml'), **config)
    best = YOLO(str(OUT / 'runs/blood_cells/weights/best.pt'))
    state['phase'] = 'evaluating_test'
    write_state()
    results = best.val(data=str(OUT / 'data.yaml'), split='test', device=0,
                       imgsz=640, batch=16, plots=True, project=str(OUT / 'evaluation'), name='test')
    report = {'split': 'test', 'metrics': results.results_dict, 'speed_ms': results.speed,
              'classes': {str(k): v for k, v in results.names.items()},
              'per_class': [{'class_id': int(c), 'name': results.names[int(c)],
                             'precision': float(results.box.p[i]), 'recall': float(results.box.r[i]),
                             'mAP50': float(results.box.ap50[i]), 'mAP50_95': float(results.box.ap[i])}
                            for i, c in enumerate(results.box.ap_class_index)],
              'training': state, 'elapsed_seconds': time.time() - started}
    (OUT / 'performance.json').write_text(json.dumps(report, indent=2, default=float))
    samples = sorted((dataset / 'images/test').glob('*.png'))[:12]
    best.predict(source=[str(p) for p in samples], device=0, save=True,
                 project=str(OUT / 'predictions'), name='test', conf=0.25)
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(OUT / 'environment.txt').open('w'), check=True)
    state['phase'] = 'complete'
    write_state()
except Exception as error:
    state.update(phase='error', error=str(error))
    write_state()
    raise
