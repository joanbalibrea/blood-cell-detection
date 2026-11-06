# CellScope integrated application validation

Executed locally on 6 November 2026 on Apple Silicon, using the existing project environment: Python 3.14.8, PyTorch 2.15.0, torchvision 0.29.2, Ultralytics 8.4.203, FastAPI 0.162.0 and Pillow 13.0.0. No model was retrained, downloaded, substituted, committed, or pushed. Existing training outputs and the earlier reports were preserved.

## Delivered behavior

The existing FastAPI app now runs YOLO26s on a full smear and EfficientNet-B0 on valid WBC crops. Both models are loaded and warmed once per server process on CPU. Inference is serialized to limit local resource use, and the classifier processes crops in batches of up to 16. Broad counts come exclusively from YOLO. The interface provides detection-only and integrated modes, separate confidence controls, original and annotated images, zoom, a reconciled subtype breakdown, a WBC crop gallery with top-three predictions, a full detection table, and matching PNG/CSV/JSON exports.

The final server was launched through the existing **CellScope: local inference** VS Code task and [launcher](../start_cellscope.command), and left running at <http://127.0.0.1:8000>. Both models report ready. The launcher and task were retained.

## Independent model verification

- **YOLO26s:** direct loading of [models/best.pt](../models/best.pt) still fails because its binary ZIP headers are damaged. The existing loader successfully repackages the existing extracted [models/best](../models/best/) records into a temporary archive, loads those trained weights, then removes the archive. The source files are not modified. Health, UI and exported model metadata explicitly identify this recovery source. A fresh binary copy of the trained archive is needed for direct loading.
- The loaded detector's actual class mapping is **0 WBC, 1 RBC, 2 Platelets**. Its training metadata records YOLO26s, 100 epochs and image size **640**. The actual head reports `end2end=False`. Existing Ultralytics postprocessing is retained; no custom NMS is added.
- **EfficientNet-B0:** [bloodmnist_classifier/models/best.pt](../bloodmnist_classifier/models/best.pt) loads directly with `weights_only=True`; every parameter matches strictly. Checkpoint metadata records **best epoch 25**, validation macro F1 **0.9973592103606062**, and a configured 40 epochs. The existing saved evaluation records completed training at 40 epochs. The eight-output classifier has 4,017,796 parameters.
- Verified class order: **basophil, eosinophil, erythroblast, immature_granulocytes, lymphocyte, monocyte, neutrophil, platelet**. This is read and checked against checkpoint metadata and the training code, not inferred from folder sorting.
- Inference reuses the actual validation transform: **RGB → bicubic resize short edge 256 → center crop 224 × 224 → tensor → normalize mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]**. Verified output tensor shape: **3 × 224 × 224**. A known-color normalization check and exact tensor equality against the training transform passed.
- Independent classifier inference on `../bloodmnist/train/basophil/train_01099.png` produced a basophil prediction with eight finite normalized probabilities. This training-image smoke check is not an accuracy evaluation. The sibling dataset path was resolved correctly.

SHA-256 hashes were measured before and after implementation and matched:

| Original checkpoint | SHA-256 |
| --- | --- |
| YOLO supplied archive | `45623f727ca19d05631739be2b38b36e954ae5709b09908c884e4cce1e77f635` |
| EfficientNet-B0 | `38c9c941364204ea01ccf7e101897c99e143e774aefeb44880047dd3ff73d514` |
| Teacher ResNet18, not loaded | `b33b671e6173205214e505c0956780fc48beed045ba5a75e6fa77149b178d29f` |

## Real integrated inference

Seven held-out detection images were run through both real models with YOLO threshold **0.25**, subtype threshold **0.70**, and square-crop padding **0.12**. Each happened to contain one detected WBC. No dataset labels were used as predictions. The subtype column below is the model output, not verified truth.

| TXL-PBC test image | RBC | WBC | Platelets | EfficientNet prediction | Classifier score | Status |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `03967e3a34d05f4583e6d361f2c333a0.png` | 19 | 1 | 1 | neutrophil | 0.999999 | predicted |
| `d0410b5b65fd577885b711e70e21c3e9.png` | 15 | 1 | 0 | eosinophil | 0.992124 | predicted |
| `00552f60c43350a0bf516cfbc13db058.png` | 12 | 1 | 0 | eosinophil | 0.914130 | predicted |
| `3d62c85fc14657b49c8c4c6572b2f5d4.png` | 9 | 1 | 0 | immature_granulocytes | 0.679870 | uncertain |
| `66c07dae05865491b56b16bccef676bf.png` | 16 | 1 | 0 | eosinophil | 0.978468 | predicted |
| `aab3e94d732651b0917cc2c7878b12d5.png` | 12 | 1 | 1 | basophil | 0.999964 | predicted |
| `cee3bdf7b7485d84a1f3e27953f4fa00.png` | 31 | 1 | 0 | immature_granulocytes | 0.988936 | predicted |

Measured CPU pipeline time was **239.5–280.2 ms** for these particular small images, excluding request transfer and JSON serialization. This is not a throughput benchmark or a performance promise for large uploads. The first example's WBC detector score was **0.958912**, separately from classifier score **0.9999985695**. No joint probability is calculated.

The seven crop previews and representative annotated image were visually inspected. The inspected crops include full nucleated cells with surrounding context; stain, nucleus appearance, scale and image resolution visibly vary. This review confirms crop alignment and rendering only. Several confident subtype assignments still require expert verification. One naturally low-confidence output was correctly separated as uncertain. No model disagreement occurred naturally in this small sample; explicit policy tests cover both non-leukocyte outputs.

Local verification outputs, including the crop contact sheet, representative PNG/CSV/JSON and machine-readable evidence, are in [cellscope_app/test_artifacts/integrated](../cellscope_app/test_artifacts/integrated/), which remains ignored by Git.

## Automated and browser checks

The [existing API tests](../cellscope_app/tests/test_api.py) and [new integration tests](../cellscope_app/tests/test_integrated.py) completed **20 tests successfully** in **2.670 seconds** on the final backend:

```sh
cd "/Users/bali/Documents/Master UIC/M4 - Advanced AI/Blood cells project"
.venv/bin/python -m unittest discover -s cellscope_app/tests -v
```

Coverage includes real detection/classification, verified class mapping, exact preprocessing, single-versus-batch probabilities, a technical montage with multiple real WBC crops classified in one batch, image dimensions and clipped coordinates, PNG decoding, every CSV row and eight-class probability, detection-only compatibility, empty/invalid/outside/nonfinite crops, bounded memory for extreme aspect ratios, subtype reconciliation, real uncertainty, no-WBC images, invalid request settings, missing models and explicit per-WBC classification failures. Disagreement tests use clearly synthetic probability fixtures only to verify review policy: predictions and all probabilities remain intact, disagreement takes precedence over uncertainty in the exclusive breakdown, and each WBC is counted once.

Additional checks:

| Check | Observed result |
| --- | --- |
| Offline operation | Both networks loaded and ran all seven smears with socket connection methods blocked: **zero connection attempts** |
| Dependency consistency | `pip check`: **No broken requirements found**; no package installation or environment replacement was needed |
| Browser upload | File chooser, preview, real analysis, and drag-and-drop handler exercised successfully |
| Browser results | 21 objects, 19 RBC, 1 WBC and 1 platelet; neutrophil gallery and one reconciled subtype assignment |
| Mode change | Clears old results/exports; detection-only retains broad counts and accounts for WBC as not classified |
| Uncertainty control | Raising minimum subtype confidence to 1 produces a visible uncertain status and warning while retaining the prediction and broad counts |
| Crop inspection | Gallery crop opens in the inspection dialog; zoom reached 300% |
| Clear/reset | Clears counts and exports while keeping the selected image preview |
| PNG export | Browser button generated a valid PNG with the standard eight-byte signature and matching report annotation |
| CSV export | Browser button generated 21 detection rows; CSV text exactly matches the JSON report's authoritative CSV |
| JSON export | Complete structured report, including separate classifier metadata and all 21 detections, generated by its browser button |
| Responsive layout | At 390 px viewport width, no horizontal document overflow; table scrolls within its panel |
| Frontend runtime | No JavaScript errors during the final upload/reset/drop/analysis flow |
| Editor diagnostics | No reported errors in application files |
| Privacy/security | Existing 12 MB/16 MP limits, image decoding checks, EXIF handling, JPEG/PNG restrictions and private static-directory routing pass the original API tests |

Browser-generated download payloads and links were verified. Saving to a normal browser's chosen download folder was not independently confirmed in the integrated preview; Safari or Chrome can use the same local URL if the preview suppresses file saves. The test client emits an installed-Starlette deprecation notice about HTTPX; all tests and normal app requests pass.

## Scientific limits

BloodMNIST's previously saved **99.18% test accuracy / 99.28% macro F1** describe isolated BloodMNIST images. No new BloodMNIST evaluation was performed and these metrics do not measure the integrated pipeline. WBC subtype reference labels were not available for this crop evaluation, so end-to-end subtype accuracy is **not established**. Detector mistakes, missed cells, stain differences, crop framing, scale and resolution affect results. High confidence can accompany an incorrect subtype. Thresholds support review and are not clinically validated operating points. The application is for education and research and is not clinically validated.

## Files changed

- Added [classification.py](../cellscope_app/classification.py): strict offline EfficientNet loading, metadata checks and batched CPU inference using training transforms.
- Added [pipeline.py](../cellscope_app/pipeline.py): bounded square crops, combined results, preserved predictions, review policy, reconciliation and authoritative CSV generation.
- Updated [inference.py](../cellscope_app/inference.py): retain existing detector loading/recovery/postprocessing; verify exact class IDs and share numbered combined annotation rendering.
- Updated [main.py](../cellscope_app/main.py): startup loading of both networks, health metadata, mode/threshold/padding options and bounded analysis concurrency.
- Updated [index.html](../cellscope_app/static/index.html), [app.js](../cellscope_app/static/app.js) and [style.css](../cellscope_app/static/style.css): integrated dark UI, controls, subtype cards, gallery, detailed table and matching exports.
- Updated [test_api.py](../cellscope_app/tests/test_api.py) for the new application title; added [test_integrated.py](../cellscope_app/tests/test_integrated.py).
- Updated [requirements-app.txt](../requirements-app.txt) to document the already installed PyTorch/torchvision versions, and [README.md](../README.md) with pipeline, usage, interpretation, API, tests and limitations.
- Added this report. Existing launcher, task, training code, model weights, datasets, training dashboards, previous reports and teacher baseline remain unchanged.

Git commands could not run because this Mac's command-line developer tools are unavailable. No repository was initialized and no staging, commit or push operation was attempted. The existing Git index timestamp remained unchanged throughout this work.

## Reopen, stop and restart

```sh
zsh "/Users/bali/Documents/Master UIC/M4 - Advanced AI/Blood cells project/start_cellscope.command"
```

Open <http://127.0.0.1:8000>. Use the currently running server first. To stop it, press **Control+C** in the **CellScope: local inference** terminal. To restart, run the same launcher command. Health is at <http://127.0.0.1:8000/api/health> and interactive API documentation is at <http://127.0.0.1:8000/docs>.
