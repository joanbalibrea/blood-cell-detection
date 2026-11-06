# Blood cell detection with YOLO26

Private Kaggle GPU training, dataset auditing, held-out evaluation and an HTML monitoring dashboard for TXL-PBC. Classes: WBC, RBC and Platelets. The source dataset lives outside this repository at `../TXL-PBC`.

**Current status:** a trained YOLO26 checkpoint is now supplied locally, with 100 epochs recorded in its metadata. The new local inference application is described below. Some historical reports still describe earlier failed Kaggle attempts; those reports are not a description of the current inference application's status or a new accuracy evaluation.

## CellScope — local blood cell detection

The FastAPI application in [cellscope_app](cellscope_app/) detects **WBC, RBC, and Platelets** using your existing trained checkpoint. It uses CPU, runs on Apple Silicon, and needs no Kaggle, CUDA, API key, or internet after setup. The existing [training dashboard](dashboard/) remains separate.

### Start the application

Open Terminal and paste this single command (works on another day and after restarting VS Code):

```sh
zsh "/Users/bali/Documents/Master UIC/M4 - Advanced AI/Blood cells project/start_cellscope.command"
```

Keep the Terminal window open. Once it says “Application startup complete”, open **<http://127.0.0.1:8000>** in your browser. You can also double-click [start_cellscope.command](start_cellscope.command) in Finder, or run the **CellScope: local inference** VS Code task. If port 8000 is already occupied by CellScope, use the existing page; otherwise stop the other server first.

To stop: click the Terminal window and press **Control+C**. To restart: run the same command again. The launcher finds this project from its own location, even if your current Terminal folder is elsewhere.

### First-time setup or reinstall

The existing `.venv` is configured and the application dependencies are installed. If rebuilding that environment, use Python 3.11 or later with PyTorch support for your Mac (this installation uses Python 3.14):

```sh
cd "/Users/bali/Documents/Master UIC/M4 - Advanced AI/Blood cells project"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
```

Only setup requires internet. [requirements-app.txt](requirements-app.txt) is separate from the existing training/tooling [requirements.txt](requirements.txt). Do not install CUDA packages for this app.

### Analyze and export

1. Browse for, or drop, a JPG/JPEG/PNG microscopy image into the upload area. Held-out examples are in `../TXL-PBC/images/test/`. Limits: **12 MB**, **16 megapixels**, one still image at a time. Transparent pixels are composited on white and JPEG orientation is respected.
2. Leave confidence at **0.25** initially and click **Detect Cells**. All detections and counts come from the trained model, never from dataset labels.
3. Compare the original and annotated images. **RBC = orange**, **WBC = purple**, **Platelets = green**. Labels show the predicted class and actual model confidence. Click either image for a larger view with zoom and scrolling.
4. Adjust confidence from **0.05–0.95**, then click **Detect Cells** again. This clears the previous result and reruns inference. A higher threshold usually retains fewer predictions. Model scores are not calibrated clinical probabilities.
5. Review counts and sort the detection table by class or confidence. Coordinates are floating-point pixels in the original image after EXIF orientation, reported as `(x_min, y_min, x_max, y_max)`. The table rounds for readability; exports preserve numeric precision.
6. Use **PNG** for the annotated image, **CSV** for every detection, or **JSON** for the summary and detections. Downloads match the displayed run. If the embedded VS Code preview does not save a file, open the same local URL in Safari or Chrome. **Clear results** keeps the image; **×** removes it so you can choose another.

Counts are **objects detected in this image**, not blood-cell concentrations. The model does not identify white-cell subtypes. This is an educational/research tool, not a clinically validated diagnostic application. Overlapping cells may have overlapping boxes; detections can still be missed or mistaken.

### Checkpoint integrity and recovery

The app first loads **[models/best.pt](models/best.pt)**. During setup that file's ZIP headers were unreadable by PyTorch and contained text replacement bytes. The existing extracted **`models/best/`** checkpoint was intact: its class mapping was `{0: 'WBC', 1: 'RBC', 2: 'Platelets'}` and its training metadata recorded 100 epochs at image size 640.

When the main archive is unreadable, the loader can repackage those **existing extracted checkpoint records**, byte for byte, into a temporary archive. It loads the same supplied trained model, then deletes the temporary archive. A visible banner and `/api/health` report this recovery source. Neither original model source is changed. No official/pretrained replacement is downloaded or used.

For direct loading, restore a fresh **binary** copy of your trained `best.pt` to `models/best.pt` and restart the server. Do not paste binary weights through a text editor. If neither copy loads, the app reports model unavailable and refuses inference rather than inventing results.

The supplied checkpoint's detection head reports `end2end=False`. Inference uses Ultralytics' own postprocessing without overriding NMS or adding duplicate suppression. The original training image size (640) is read from checkpoint metadata. CPU is the supported and tested device; MPS is not enabled.

### API, privacy, and tests

- `GET /api/health`: readiness, actual class mapping, CPU, model source, image size, recovery warning.
- `POST /api/analyze?confidence=0.25`: multipart upload with one `file`; returns original dimensions, counts, class IDs/names, exact confidence scores, pixel boxes, timing, and an annotated PNG data URI.
- Interactive API documentation: <http://127.0.0.1:8000/docs>.
- Invalid image/empty upload: 400; unsupported extension: 415; oversized upload: 413; invalid confidence: 422; unavailable model: 503.

Only the app's dedicated static folder is served. The project root, model files, datasets, `.env`, and training dashboard are not exposed. The server binds to this computer's loopback interface. Uploads/results are not kept in an application database or saved to the project; multipart temporary files are closed after each request. Results remain in browser memory until cleared or the page closes; downloaded files remain wherever your browser saves them. This page uses no external fonts, scripts, or services.

Run the integration checks (they load the real local checkpoint and use held-out images):

```sh
cd "/Users/bali/Documents/Master UIC/M4 - Advanced AI/Blood cells project"
# Only if test dependencies are missing:
# .venv/bin/python -m pip install -r requirements-app-dev.txt
.venv/bin/python -m unittest cellscope_app.tests.test_api -v
```

These functional checks do not constitute a new precision/recall or clinical validation study.

See the [local application validation report](reports/cellscope_validation.md) for executed checks, checkpoint recovery details, and verification limits.

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

Initialized on branch `main`, with commit and push. Secrets, the local environment, dataset staging and model files are ignored. macOS command-line Git requires developer tools on this machine; the installed Dulwich package can inspect the repository in the meantime:

```sh
python -m dulwich status
```

Commit and GitHub push require the project owner's approval.
