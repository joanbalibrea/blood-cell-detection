# Blood cell detection with YOLO26

Private Kaggle GPU training, dataset auditing, held-out evaluation and an HTML monitoring dashboard for TXL-PBC. Classes: WBC, RBC and Platelets. The source dataset lives outside this repository at `../TXL-PBC`.

**Current status:** a trained YOLO26 checkpoint is now supplied locally, with 100 epochs recorded in its metadata. The new local inference application is described below. Some historical reports still describe earlier failed Kaggle attempts; those reports are not a description of the current inference application's status or a new accuracy evaluation.

## CellScope — Integrated Blood Cell Analysis

The FastAPI application in [cellscope_app](cellscope_app/) combines **YOLO26s detection** with **EfficientNet-B0 single-cell classification**. It loads both trained models once, uses CPU on Apple Silicon, and needs no Kaggle, CUDA, API key, or internet after setup. The existing [training dashboard](dashboard/) remains separate.

### Two-stage pipeline and required checkpoints

1. [models/best.pt](models/best.pt): YOLO26s detects **0 WBC, 1 RBC, 2 Platelets** on the uploaded smear at training image size 640. Ultralytics supplies the model-specific postprocessing; the app adds no NMS controls. All broad counts come from these detections.
2. [bloodmnist_classifier/models/best.pt](bloodmnist_classifier/models/best.pt): EfficientNet-B0 classifies only valid WBC crops, in batches of up to 16. The model is constructed without downloading pretrained weights, then loads the checkpoint state strictly. Verified metadata records **best epoch 25**, with a configured 40-epoch experiment.

The verified classifier index order is **basophil, eosinophil, erythroblast, immature_granulocytes, lymphocyte, monocyte, neutrophil, platelet**. Inference reuses [training/common.py](bloodmnist_classifier/training/common.py): RGB, bicubic short-edge resize to **256**, center crop **224 × 224**, tensor conversion, mean **[0.485, 0.456, 0.406]**, standard deviation **[0.229, 0.224, 0.225]**. There is no inference augmentation.

WBCs are cropped from the original image after EXIF orientation. The pipeline clips source coordinates, extends to square context with **12% padding per side** by default (configurable 0–25%), and fills missing edge context white. A square avoids shape distortion and default padding accommodates the validation center crop. Large square crops are downsampled to at most 1024 pixels before the exact training transform to bound memory; this is recorded in crop metadata. Gallery previews are at most 192 pixels. Detection boxes and counts never change during cropping.

The teacher's [ResNet18 checkpoint](bloodmnist_classifier/models/resnet18-bloodmnist-wbc.pt) is not loaded or changed. BloodMNIST lives at [../bloodmnist](../bloodmnist/), beside this project; it is not needed to use the finished application. Representative detection images live at [../TXL-PBC/images/test](../TXL-PBC/images/test/).

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
2. Choose **Detection + WBC classification** (default), leave YOLO confidence at **0.25** and minimum subtype confidence at **0.70**, and click **Analyze Blood Smear**. **Detection only** preserves the original single-model workflow and remains available if the classifier cannot load.
3. Compare original and annotated images. **RBC = orange**, **WBC = purple**, **Platelets = green**. Numbered labels use **D** for YOLO detection confidence and **C** for EfficientNet confidence; WBC labels include the unmodified classifier prediction and a review marker when needed. Click either image or a gallery crop for zoom up to 300%.
4. Review the six leukocyte subtype counts and the WBC gallery, including each crop's top three predictions and separate detector/classifier scores. The table includes every detection, both predictions, exact original-image coordinates, and review status. Sort by broad class or detector score. The table rounds scores and coordinates; exports retain precision.
5. Adjust YOLO confidence **0.05–0.95**, subtype confidence **0–1**, or crop padding, then analyze again. Changing settings clears the previous run and disables its exports so displayed results and downloads stay consistent.
6. Use **PNG** for the combined annotated image, **CSV** for every object including all eight classifier probabilities on WBCs, or **JSON** for the complete structured report including crop previews, annotation, model/preprocessing metadata, timings, thresholds and review flags. Downloads match the displayed run. If the embedded preview suppresses saves, use Safari or Chrome. **Clear results** keeps the selected image; **×** removes it.

### Interpretation and scientific limitations

Counts are **objects detected in this image**, not blood-cell concentrations. Classification never adds or subtracts a YOLO detection. Every YOLO WBC enters exactly one category: a confident leukocyte subtype, **uncertain**, **model disagreement**, **could not classify**, or **not classified** in detection-only mode. Their sum equals the WBC count. An erythroblast or platelet prediction on a YOLO WBC is retained, including all probabilities, and always flagged as model disagreement. Disagreement takes precedence in the exclusive breakdown; `low_confidence` remains independently recorded, so low-confidence disagreements are not counted twice. A failed crop or inference batch remains visible as a failure and does not disappear from the count.

Detector confidence and classifier confidence are independent scores. They are never multiplied into a joint probability. A high softmax score is not proof of correctness. The historical BloodMNIST test accuracy **99.18%** and macro F1 **99.28%** apply to isolated BloodMNIST images, not YOLO crops. Domain shift in stain, scale, crop framing and resolution can cause confident mistakes. The detector can also miss cells or produce incorrect broad classes, which the WBC-only second stage cannot recover. No labeled subtype reference set for these YOLO crops has been evaluated, and no integrated subtype accuracy is claimed. This is an educational/research tool, not a clinically validated diagnostic application.

### Checkpoint integrity and recovery

The app first loads **[models/best.pt](models/best.pt)**. During setup that file's ZIP headers were unreadable by PyTorch and contained text replacement bytes. The existing extracted **`models/best/`** checkpoint was intact: its class mapping was `{0: 'WBC', 1: 'RBC', 2: 'Platelets'}` and its training metadata recorded 100 epochs at image size 640.

When the main archive is unreadable, the loader can repackage those **existing extracted checkpoint records**, byte for byte, into a temporary archive. It loads the same supplied trained model, then deletes the temporary archive. A visible banner and `/api/health` report this recovery source. Neither original model source is changed. No official/pretrained replacement is downloaded or used.

For direct loading, restore a fresh **binary** copy of your trained `best.pt` to `models/best.pt` and restart the server. Do not paste binary weights through a text editor. If neither copy loads, the app reports model unavailable and refuses inference rather than inventing results.

The supplied checkpoint's detection head reports `end2end=False`. Inference uses Ultralytics' own postprocessing without overriding NMS or adding duplicate suppression. The original training image size (640) is read from checkpoint metadata. CPU is the supported and tested device; MPS is not enabled.

### API, privacy, and tests

- `GET /api/health`: original detector status plus `classifier` metadata and `integrated_ready`.
- `POST /api/analyze?mode=integrated&confidence=0.25&subtype_confidence=0.70&padding=0.12`: multipart upload with one `file`. Returns broad counts/detections, separate `classification` objects on WBCs, full eight-class probabilities, crop previews and metadata, `wbc_breakdown`, PNG data URI, and authoritative CSV text. Existing top-level detector fields remain available.
- Omitting `mode` defaults to **detection** for existing API clients; the web UI explicitly requests **integrated**. `classification.status` is `not_applicable` for RBCs and platelets and `not_run` for unclassified WBCs in detection-only mode.
- Interactive API documentation: <http://127.0.0.1:8000/docs>.
- Invalid image/empty upload: 400; unsupported extension: 415; oversized upload: 413; invalid mode/threshold/padding: 422; unavailable required model: 503; another analysis in progress: 429. Partial classification failures return detections with explicit failed WBC statuses.

Only the app's dedicated static folder is served. The project root, model files, datasets, `.env`, and training dashboard are not exposed. The server binds to this computer's loopback interface. Uploads/results are not kept in an application database or saved to the project; multipart temporary files are closed after each request. Results remain in browser memory until cleared or the page closes; downloaded files remain wherever your browser saves them. This page uses no external fonts, scripts, or services.

Run the integration checks (they load the real local checkpoint and use held-out images):

```sh
cd "/Users/bali/Documents/Master UIC/M4 - Advanced AI/Blood cells project"
# Only if test dependencies are missing:
# .venv/bin/python -m pip install -r requirements-app-dev.txt
.venv/bin/python -m unittest discover -s cellscope_app/tests -v
```

These functional checks do not constitute a new precision/recall or clinical validation study.

See the [integrated application validation report](reports/cellscope_integrated_validation.md) for the two-model checks and scientific limits. The earlier [detection-only validation report](reports/cellscope_validation.md) is preserved.

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
