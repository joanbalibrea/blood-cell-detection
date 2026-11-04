# Blood cell detection with YOLO26

Private Kaggle GPU training, dataset auditing, held-out evaluation and an HTML monitoring dashboard for TXL-PBC. Classes: WBC, RBC and Platelets. The source dataset lives outside this repository at `../TXL-PBC`.

**Current status:** dataset preparation and local checks passed, but Kaggle returned a CPU runtime despite GPU requests. Training is blocked; no fine-tuned model or measured accuracy is available. See the [execution report](reports/execution.md) for evidence and the required Kaggle account/runtime check.

## Setup

Use Python 3.11 or later for local tooling. The current local environment uses Python 3.14; model training runs in Kaggle's CUDA image.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put `KAGGLE_API_TOKEN` in `.env`, following [.env.example](.env.example). The actual secret file is ignored by Git. Never place credentials in the training entrypoint, dashboard or dataset upload folder.

## Workflow

```sh
python audit_dataset.py
python manage.py prepare
python manage.py upload
python bundle_offline.py
python manage.py launch
python manage.py watch
```

The dataset and kernel are private. `prepare` converts PNGs to RGB in an ignored staging folder. `upload` creates a new dataset and should be run only once for this slug. `bundle_offline.py` downloads Linux wheels and official model weights, creates a private assets dataset, and attaches it to kernel metadata; use `--update` to version an existing asset bundle. Local internet access is needed for this step. Training uses no internet and relies on Kaggle's installed CUDA/PyTorch environment.

`launch` starts a new kernel version and consumes Kaggle GPU quota; do not run it just to check status. Use `sync` for a single update or `watch` for updates every 45 seconds. The watcher stops after completion/error/cancellation and retrieves available output files. Kaggle account verification and GPU availability are required.

The already submitted kernel is recorded in [run.json](run.json). Existing runs can be checked without uploading or launching again:

```sh
python manage.py sync
```

## Dashboard

In a separate terminal:

```sh
python -m http.server 8765 --bind 127.0.0.1 --directory dashboard
```

Open <http://127.0.0.1:8765>. Serve only the dashboard directory so `.env` is not exposed. The page refreshes every 15 seconds. Job status comes from the local watcher. To feed live epoch callbacks and training loss into the dashboard, run this in another terminal:

```sh
python stream_logs.py
```

The stream reconnects if it ends before the job finishes. Completed training history replaces streamed metrics when Kaggle releases saved outputs. No simulated performance values are displayed.

## Reports and model

- [Dataset report](reports/dataset.md) and [machine-readable audit](reports/dataset.json).
- [Training protocol](reports/training.md).
- After successful completion: `reports/performance.md`, `reports/performance.json`, `reports/training_history.csv`.
- Downloaded model: `outputs/runs/blood_cells/weights/best.pt`.
- Kaggle console, plots, checkpoints and predictions are retained in ignored `outputs/`.

Inference in an environment with `ultralytics==8.4.32` installed:

```sh
yolo predict model=outputs/runs/blood_cells/weights/best.pt source=/path/to/smear.png conf=0.25
```

These are research detections, with no clinical validation performed. Dataset rights remain with their original owners. Review the Ultralytics AGPL-3.0/Enterprise licensing terms before distribution or commercial integration.

## Git

Initialized on branch `main`, with no commit and no remote push. Secrets, the local environment, dataset staging and model files are ignored. macOS command-line Git requires developer tools on this machine; the installed Dulwich package can inspect the repository in the meantime:

```sh
python -m dulwich status
```

Commit and GitHub push require the project owner's approval.
