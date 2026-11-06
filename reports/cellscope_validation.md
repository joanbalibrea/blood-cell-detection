# CellScope local application validation

Executed on 6 November 2026, on this Apple Silicon Mac, using the existing Python 3.14 environment, PyTorch 2.15.0 and Ultralytics 8.4.203. No retraining, Kaggle execution, model download, commit, or push was performed.

## Result

The local FastAPI application starts at <http://127.0.0.1:8000>, loads the existing trained checkpoint once per server process, and performs real CPU inference. Its frontend was opened and exercised in the integrated browser. The final server was launched through the same [launcher](../start_cellscope.command) documented for everyday use and left running.

## Checkpoint provenance and recovery

- Directly loading [models/best.pt](../models/best.pt) failed: `PytorchStreamReader ... unsupported multidisk archive`. ZIP inspection also failed and showed UTF-8 replacement bytes in binary headers. The file was not rewritten.
- SHA-256 of that supplied archive: `45623f727ca19d05631739be2b38b36e954ae5709b09908c884e4cce1e77f635`.
- The existing extracted `models/best/` records were readable. Repackaging those exact files into a temporary PyTorch archive successfully loaded the supplied fine-tuned model. The temporary archive is removed after application loading. Neither original model source is altered.
- Actual classes: **0 WBC, 1 RBC, 2 Platelets**. Metadata records YOLO26s training for 100 epochs, training image size 640. The final detection head reports `end2end=False`.
- The app attempts the primary archive first. If recovery is needed, the UI banner, health endpoint and report identify the extracted source. No other model is substituted. Restoring a fresh binary copy of the trained archive and restarting will use that file directly.

## Executed checks

| Check | Observation |
| --- | --- |
| Server startup | Uvicorn started on loopback port 8000; model warmed on CPU |
| Initial model smoke test | Held-out `d0410b5b65fd577885b711e70e21c3e9.png`: 16 detections at confidence 0.25; rendered boxes inspected |
| All-class held-out example | `03967e3a34d05f4583e6d361f2c333a0.png`, 360 × 363 px: **19 RBC, 1 WBC, 1 platelet; 21 total**, at 0.25 |
| Higher threshold | Same image at 0.95: **7 RBC, 1 WBC, 0 platelets; 8 total**, tested through browser controls and API |
| Counts and boxes | Counts equal returned detection classes; scores meet the requested threshold; pixel boxes lie within original dimensions |
| Annotation | PNG decodes at original dimensions, includes orange/purple/green class colors; visible boxes align with cells in the inspected sample |
| Frontend | File chooser, preview, detect, loading state, class sorting, 300% inspection zoom, clear results and remove image exercised |
| Responsive layout | At 390 px viewport width, no horizontal document overflow |
| PNG export | Button-generated data validated: `image/png`, valid eight-byte PNG signature, 123,627 bytes for the representative run |
| CSV export | Button-generated payload: header plus 21 detection rows, exact class IDs, scores and coordinates; 2,176 bytes |
| JSON export | Button-generated report: 21 detections and matching counts, dimensions, threshold and source metadata; 7,063 bytes for that run |
| Invalid uploads | Empty bytes, fake PNG, unsupported extension and GIF disguised as PNG rejected with helpful 400/415 responses |
| Size limits | Multipart and streamed oversized bodies rejected with 413; images over 16 megapixels rejected |
| Input formats | JPEG and RGBA PNG inference tested; EXIF rotation and transparent-pixel compositing verified |
| Invalid confidence | Out-of-range, nonnumeric and NaN confidence rejected with 422 |
| Missing model behavior | Unavailable-model path returns 503 with a restart message |
| Offline model operation | Startup and real CPU inference with socket connection methods disabled: **zero network attempts**, same 21 detections |
| Private files | Requests for project credentials path, model, README and training dashboard through the app return 404; static app responses use `no-store` |
| Dependency consistency | `pip check`: no broken requirements |
| Training dashboard | Its five tracked files byte-match Git's existing index; dataset and training files were not edited |
| Editor diagnostics | No reported errors in the application or VS Code task configuration |

The [integration suite](../cellscope_app/tests/test_api.py) completed **9 tests successfully** with `unittest`. VS Code had no discovered tests for that file, so the documented terminal command was used. A Starlette deprecation warning about its current HTTPX test client does not affect the passing checks or normal application operation.

## Verification limits

- Browser export contents and generated download links were verified. The integrated browser did not emit a download completion event, so saving files to a normal browser's chosen download directory was not confirmed. Use Safari or Chrome if the embedded preview suppresses saves.
- CPU is the supported device. MPS/CUDA acceleration was not enabled or tested.
- Functional checks and visual box inspection are not a new model accuracy evaluation. Predictions can contain false positives, missed cells and overlapping boxes. Counts must not be interpreted as clinical blood-cell concentrations.
- Earlier training/performance reports were preserved as historical records. This report does not replace them with invented metrics.

## Files added or modified

- [Application entrypoint](../cellscope_app/main.py), [model and image handling](../cellscope_app/inference.py), package initializers and [integration tests](../cellscope_app/tests/test_api.py).
- [HTML interface](../cellscope_app/static/index.html), [dark-theme styles](../cellscope_app/static/style.css), [browser interactions and exports](../cellscope_app/static/app.js).
- [Application dependencies](../requirements-app.txt), [optional test dependencies](../requirements-app-dev.txt), executable [launcher](../start_cellscope.command).
- [README](../README.md), [ignore rules](../.gitignore), and this report.
- Workspace-level [VS Code launch task](../../.vscode/tasks.json).

Original weights, dataset, training dashboard and training scripts remain untouched. Existing staged Git changes were left as found; nothing was committed or pushed.
