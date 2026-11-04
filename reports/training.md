# Training protocol

## Model and task

Fine-tune pretrained Ultralytics **YOLO26s** (`yolo26s.pt`) for bounding-box detection of **WBC, RBC, and Platelets**, with class IDs 0, 1, and 2. This dataset cannot train WBC subtype recognition. Ultralytics is pinned to **8.4.32** in the Kaggle entrypoint.

Official model documentation: <https://docs.ultralytics.com/models/yolo26/>.

## Data handling

The original 882/252/126 train/validation/test image split is preserved. RGB conversion removes alpha channels without resizing or changing box coordinates. Source images and labels are not modified. The private Kaggle dataset contains only converted images, labels, and a data configuration; credentials are never uploaded.

See [dataset audit](dataset.md) for actual class counts and validation findings. Exact SHA-256 matching found no cross-split duplicates; this does not rule out related images, near duplicates, or shared patients. Patient identifiers are unavailable.

## Configuration

| Setting | Value |
|---|---|
| Pretrained model | YOLO26s |
| Epoch limit | 100 |
| Early stopping patience | 20 epochs |
| Image size | 640 pixels |
| Batch size | 16 |
| Seed | 42 |
| Accelerator | Kaggle GPU, CUDA device 0 |
| Data loader workers | 2 |
| Automatic mixed precision | Enabled |
| Deterministic mode | Enabled |
| Mosaic closing | Final 10 epochs |
| Checkpoint interval | 10 epochs |
| Maximum Kaggle session | 6 hours |

Other optimizer, learning-rate, and augmentation settings use the pinned Ultralytics defaults. The run saves resolved arguments, package versions, GPU identity, loss history, validation curves, and checkpoints. GPU identity and measured runtime are reported after execution; deterministic settings do not guarantee identical results across GPU/software versions.

## Evaluation

Ultralytics selects best weights using validation fitness during training. Only those weights are evaluated on the test split after training. Report overall and per-class precision, recall, mAP at IoU 0.50 and mAP averaged over IoU 0.50–0.95, plus inference timing, confusion matrices, PR curves, and 12 annotated test predictions. Model validation uses Ultralytics default confidence/IoU settings; preview predictions use confidence 0.25.

Precision and recall are Ultralytics' operating-point summaries; they are distinct from average precision across confidence thresholds. RBCs dominate the data, so per-class measurements are essential. Test results are not used for model tuning. No measured accuracy is claimed until the job completes.

## Outputs

- `runs/blood_cells/weights/best.pt` and `last.pt`: model weights.
- `runs/blood_cells/results.csv`: epoch loss and validation metrics.
- `evaluation/test/`: final test plots.
- `performance.json`: measured aggregate/per-class metrics and timings.
- `progress.json`: final epoch, phase, GPU and runtime metadata.
- `environment.txt`: Kaggle package versions.
- `predictions/test/`: example detections.

All these paths are relative to downloaded Kaggle outputs. `manage.py sync` copies report data into the local dashboard after Kaggle exposes saved outputs. `stream_logs.py` consumes Kaggle's console stream and publishes actual per-epoch callback metrics to the dashboard; the Kaggle run page also provides its live console.

The training session installs pinned packages from a private offline bundle. YOLO26s supplies the pretrained model, and a separate YOLO26n checkpoint is used only for Ultralytics' AMP startup check. Asset SHA-256 digests are recorded in [offline-assets.json](offline-assets.json). The first submission failed before training because Kaggle could not resolve PyPI; the offline bundle removes that network dependency.
