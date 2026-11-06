# CellScope — Blood Cell Classification

An independent eight-class BloodMNIST classifier for Module 4, Advanced Topics in AI for Healthcare. It assigns one label to an image of an individual cell. It does not find cells in a whole smear. The existing YOLO project and inference application are separate and have not been integrated here.

## Actual execution status

**All 40 epochs are complete and verified. The trained checkpoint, full history, evaluation reports, confusion matrices and example predictions are downloaded. Local CPU inference with the actual trained checkpoint passed.** The original Kaggle run stopped early at epoch 33; a state-restored continuation completed epochs 34–40 at the user's request. See the [continuation protocol](continuation/README.md).

The validation-selected best model remains epoch **25**. Its unchanged weights achieved **99.18% test accuracy**, **99.28% macro F1**, and **0.999664 macro one-vs-rest AUROC** on 3,421 images. Since the selected weights did not change during continuation, the original test predictions were reused: only one test inference pass was needed. Model selection used validation only; no checkpoint was selected using test results. The [final report](outputs/evaluation_report.md) and [machine-readable evaluation](outputs/evaluation.json) contain all measured metrics. The [dashboard](http://127.0.0.1:8766/dashboard/) displays 40/40 epochs.

The private dataset [cellscope-bloodmnist-classifier-assets](https://www.kaggle.com/datasets/joanbalibrearull/cellscope-bloodmnist-classifier-assets) was uploaded after explicit user approval. Its payload contains the 17,092 supplied images, relative paths/labels/hashes, loader policy, training configuration, and public ImageNet weights; approximately 138 MiB total. It contains no API credentials. See the [bundle inventory](reports/generated/offline_assets.json).

The first notebook version stopped before training because Kaggle automatically expanded the transport ZIP under its basename. The loader was corrected and tested against both expanded and intact-archive layouts. Version 2 of the [private classifier notebook](https://www.kaggle.com/code/joanbalibrearull/cellscope-bloodmnist-classifier) was then submitted after confirming version 1 had stopped. No duplicate running jobs were launched.

The existing YOLO [execution report](../reports/execution.md) records historical CPU-only provisioning problems. Both successful classifier runs actually verified Tesla T4 CUDA execution for model parameters and batches. The code continues to refuse full CPU training.

## Dataset and scientific limitations

The actual dataset is at `../../bloodmnist/` relative to this subproject, at the workspace root. Original files were not moved or edited. This differs from the path in the initial brief.

| Original split | Images |
|---|---:|
| Train | 11,959 |
| Validation | 1,712 |
| Test | 3,421 |
| Total | 17,092 |

All files were decoded and verified: RGB PNG, 64 × 64 pixels, zero corrupt/unexpected files, zero symlinks. The classes are basophil, eosinophil, erythroblast, immature granulocytes, lymphocyte, monocyte, neutrophil and platelet. The numeric mapping follows that order, IDs 0–7. Erythroblasts and platelets are not mature WBC subtypes.

The audit found **17 exact duplicate groups overall, nine across splits**, including one eosinophil/neutrophil label conflict. Both byte hashes and decoded RGB pixel hashes were checked. The default loader policy excludes eight overlapping training entries and one overlapping validation entry, prioritizing the test split, then validation. It leaves all original files and split assignments intact, and retains every test image. Effective sizes are 11,951 / 1,711 / 3,421. This prevents exact pixel overlap among the loaded splits. It does not establish which conflicting label is correct.

The largest training class is 2.74 times the smallest. We use ordinary cross-entropy and inspect macro F1 and per-class recall so that larger classes do not conceal poor performance on smaller classes. No class weights or repeated hyperparameter searches are used for this baseline.

Patient/donor identifiers are unavailable. Exact hashing cannot establish donor independence or exclude related/near-duplicate images. The labels are taken as supplied. No performance on full-resolution smear crops or clinical deployment is established. A model trained on this dataset requires independent validation before use in a different setting.

The [dataset report](reports/generated/dataset_report.md), [full audit JSON](reports/generated/dataset.json), [manifest](reports/generated/manifest.csv), [selected exclusions](reports/generated/selected_data_policy.json), [class chart](reports/generated/class_distribution.png), and [representative images](reports/generated/representative_cells.png) contain the evidence.

## Architecture and preprocessing

EfficientNet-B0 provides ImageNet transfer learning with a compact network and fewer operations than ResNet18 at the same input resolution. Replacing its original head with eight outputs gives **4,017,796 trainable parameters**. The compact checkpoint and CPU/MPS support suit later MacBook Air inference; latency with trained weights has not yet been measured.

The official pretrained weights use RGB, bicubic resizing to 256 pixels, a 224 × 224 center crop, and ImageNet normalization with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. Validation, test and local inference use the same deterministic preprocessing. Upsampling does not create missing biological detail.

Training alone uses a random resized crop with scale 0.85–1.0, horizontal and vertical flips, rotation up to 20°, and mild color jitter. These transformations encourage robustness to orientation and staining, but validation performance must still demonstrate their usefulness.

## Local environment and dashboard

The existing project environment already has the required packages and was used without installing or upgrading its dependencies. From the workspace root:

```sh
cd "Blood cells project/bloodmnist_classifier"
../.venv/bin/python manage.py serve --port 8766
```

Open **http://127.0.0.1:8766/dashboard/**. The server binds to this Mac only and serves the dashboard, audit reports and evaluation images. It does not serve the parent project, credentials, transport bundle or checkpoints.

Stop the server with **Ctrl+C** in its terminal. Restart with the same command. If port 8766 is busy, use `--port 8767` and open the matching URL. Restarting the dashboard never starts or restarts training. Stop `watch` or `logs` with Ctrl+C as well; that does not cancel a remote Kaggle run. To stop remote training, use Kaggle's notebook session/Run controls and confirm the session stopped there.

For a separate environment on another Mac, install a supported Python and use:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python manage.py serve --port 8766
```

Use `.venv/bin/python` instead of `../.venv/bin/python` in the commands below if using that environment. Local verification used Python 3.14.8, PyTorch 2.15.0 and torchvision 0.29.2. Reproducibility across package/hardware versions is not guaranteed; each Kaggle run saves its actual PyTorch/torchvision versions and GPU diagnostics.

## Kaggle workflow

Credentials load from the existing parent `.env` into process environment using python-dotenv. The file is not edited. Shell environment values take precedence. No credentials are embedded in notebooks, generated data or browser files. Do not paste tokens into Kaggle cells or command arguments.

The original assets remain local. One compressed transport archive is necessary to attach them to the private Kaggle notebook; Kaggle may expand the ZIP into a `bloodmnist` folder on upload, and the loader supports both layouts. The offline bundle includes verified official ImageNet weights (`7f5810bc…` SHA-256), the image manifest, policy and config, so Kaggle Internet can remain off. Dataset metadata grants no new reuse rights.

For reference, preparation/submission used these commands sequentially from this subproject. **Training is complete; do not submit another job merely to view results.**

```sh
../.venv/bin/python manage.py auth
../.venv/bin/python audit_dataset.py
../.venv/bin/python manage.py prepare
../.venv/bin/python manage.py upload
```

These have already executed. `upload` creates one private dataset. A repeated upload command checks processing status instead of creating another copy. The dataset is ready; the initial launch command was:

```sh
../.venv/bin/python manage.py launch
```

This requests a T4 accelerator on one private Kaggle notebook, saves its exact version number, and disables duplicate submissions. An ambiguous timeout also blocks automatic resubmission: inspect the Kaggle page before taking any further action. Dataset processing delays do not require a new upload. The separate `retry` command requires a confirmed failed/stopped version, downloaded diagnostics and a changed notebook source; it must only be used after fixing a verified failure.

The [standalone notebook](training/bloodmnist_training.ipynb) and [standalone script](training/kaggle_train.py) contain all training/reporting code, with no dependency on Mac paths. The generated kernel metadata requests `enable_gpu=true`, accelerator `NvidiaTeslaT4`, private access and no internet. These settings are requests, not proof of GPU allocation. The run checks CUDA before extraction, verifies every image against the audit manifest, and checks model parameters, batches and the first forward pass on CUDA. It saves/prints startup diagnostics. A CPU session exits before training.

### If Kaggle supplies a CPU runtime or disables GPUs

1. Sign in to the account `joanbalibrearull` at Kaggle. Open the submitted notebook URL shown by `launch`; choose **Edit**.
2. Expand **Settings** or **Session options**, then **Accelerator → GPU T4 x2** (or an available CUDA GPU). Complete phone/account verification if requested. If GPU is disabled, read its account/quota message; that identifies the remaining manual action. Do not keep launching versions.
3. Ensure the private classifier assets dataset is attached under **Input**, and keep **Internet off**; all assets are bundled.
4. Start an interactive session and verify CUDA in a separate diagnostic cell before committing to training:

```python
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.device_count())
assert torch.cuda.is_available(), "Enable a CUDA GPU before training"
print(torch.cuda.get_device_name(0))
```

5. Once the runtime actually has CUDA, use **Save Version → Save & Run All** once. Do not leave an interactive training session and a committed training version running simultaneously. The notebook also has its own hard CUDA guard.
6. If manually saving a new notebook version, point the local watcher at that existing version, for example:

```sh
../.venv/bin/python manage.py adopt --version 2
```

Replace 2 with Kaggle's actual new version number. `adopt` verifies an existing version and launches no job.

If the original upload itself is blocked at the account or approval level, resolve that before launching; there is no remote classifier notebook yet.

## Training protocol and monitoring

The initial budget is 40 epochs with patience 8, batch size 64, AdamW learning rate 0.0003, weight decay 0.0001, cross-entropy loss and seed 42. All layers are fine-tuned. ReduceLROnPlateau halves the learning rate after three stagnant validation macro F1 checks. Training uses CUDA mixed precision; validation and final evaluation use FP32. CUDA, NumPy, Python and loader workers are seeded, with deterministic settings where available.

Every epoch records training/validation loss and accuracy, validation macro F1, learning rate, epoch, elapsed time and GPU. Training metrics describe augmented batches; validation metrics describe deterministic held-out validation data. The highest measured validation macro F1 selects `best.pt`; early stopping uses the configured minimum meaningful improvement. `last.pt` also captures optimizer/scheduler/scaler state for inspection, but automatic training resume is not implemented. Deterministic seeds cannot guarantee bit-for-bit identity across software or hardware.

Use one terminal for status/final output synchronization:

```sh
../.venv/bin/python manage.py watch
```

It queries the recorded version every 45 seconds and downloads terminal outputs once. A one-off update is:

```sh
../.venv/bin/python manage.py sync
```

In a second terminal, attempt epoch log streaming:

```sh
../.venv/bin/python manage.py logs
```

Only actual `CELLSCOPE_METRIC` records populate live history. A stream may be delayed, unavailable or disconnected; this is not continuous GPU telemetry. If unavailable, use the Kaggle console and wait for saved outputs. The status API alone supplies no epoch metrics. The dashboard refreshes local saved files every 15 seconds and shows when logs/status were observed. Pressing Refresh does not query Kaggle or affect training. Results persist after local services stop.

The dashboard flags epoch transitions where training loss decreases and validation loss increases. That is a warning to inspect generalization, not proof of overfitting by itself. Validation drives every model-selection decision; test performance does not drive tuning.

## Final evaluation and model download

After training ends, the best checkpoint is restored and one inference pass runs over all 3,421 held-out test images. Reports include accuracy, macro precision/recall/F1, weighted F1, per-class metrics and support, raw/row-normalized confusion matrices, prediction probabilities, macro one-vs-rest AUROC where every class has positive/negative support, selected correct/error examples, and inference timing. Absent-class metrics use zero for undefined precision/recall/F1, while AUROC is unavailable without required support. The report explains these choices.

Timing separates batched FP32 forward time from total loading/transfers/probabilities time and excludes warmup. It does not measure single-image Mac latency. Softmax scores are not calibrated clinical probabilities.

After `sync`, expect:

| Location | Contents |
|---|---|
| `models/` | `best.pt`, `model_config.json`, `class_mapping.json` |
| `outputs/` | History CSV/JSON, evaluation JSON/report, prediction CSV, per-class CSV, figures and selected images |
| `outputs/download/vN/` | Original Kaggle output files and logs for version N, including `last.pt` |
| `dashboard/data/` | Local dashboard snapshots of actual metrics |
| `reports/generated/` | Dataset audit, figures, manifest and offline asset inventory |

All evaluation outputs are now available. The dashboard renders the performance tables and confusion matrix. For a Markdown version open `outputs/evaluation_report.md`. Original version-2 outputs remain under `outputs/download/v2/`; the 40-epoch continuation artifacts are under `outputs/download/continuation-v1/`. The selected `models/best.pt` remains the epoch-25 model, while the continuation's `last.pt` records epoch 40. Kaggle's **Output** tab can also download files manually.

## Predicting a new single-cell image

The real trained weights have been downloaded and CPU inference has been verified:

```sh
../.venv/bin/python inference.py /absolute/path/to/cell.png
../.venv/bin/python inference.py /absolute/path/to/cell.png --device mps
```

CPU is the reliable default. MPS is optional and raises a clear error if unavailable; use CPU if a particular MPS operation is incompatible. Loading needs no internet and does not download ImageNet weights.

```python
from inference import CellClassifier

classifier = CellClassifier("models/best.pt", device="cpu")
result = classifier.predict("cell.png")
# result: predicted_class, confidence, probabilities for all eight classes
```

Reuse the object for multiple images to avoid repeated model loading. Provide a crop containing an individual cell; the detector/classifier integration is intentionally deferred. Earlier software checks used explicitly untrained temporary fixtures; final verification loaded the actual downloaded trained weights on CPU and produced eight probabilities. See [actual CPU verification](continuation/local_inference.json). That functional check is not another test-set evaluation.

## Validation and project isolation

```sh
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python manage.py bundle
```

Eight pipeline checks and five continuation checks passed. They cover metric arithmetic (including tied AUROC), missing-class behavior, duplicate exclusion, Kaggle ZIP layouts, CPU-training refusal, inference loading, report generation, fragmented log parsing, real checkpoint/optimizer/scheduler restoration, and cached-image decoding. Synthetic tests use temporary fixtures only; they never populate the dashboard or model directory. Actual T4 GPU execution, the complete 40-epoch history, and trained-weight CPU inference were separately verified from real outputs.

All implementation changes are inside this subproject. No repository was initialized and no files were committed or pushed. The subproject ignore rules cover staging archives, generated outputs, checkpoints, caches and sensitive files. Existing parent dataset ignore rules are left unchanged; review them before your own Git operations. Never commit `.env` files or dataset/weight bundles.
