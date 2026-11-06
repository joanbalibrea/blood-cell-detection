# Continue the existing run to exactly 40 epochs

The dashboard's eight-epoch display was stale. Saved Kaggle version 2 actually completed **33 epochs**, with early stopping after eight stagnant validation checks; the selected checkpoint was from epoch 25.

This continuation restores `last.pt` from epoch 33, including the model, AdamW moments, ReduceLROnPlateau scheduler, and AMP gradient scaler. It trains **epochs 34–40** with no early-stopping break. The original checkpoint was saved before the epoch-33 scheduler update, so restoration applies that missing update once; epoch 34 begins at learning rate **0.00001875**. The original checkpoint lacks RNG/worker states, so bitwise identity with an uninterrupted run cannot be guaranteed; the continuation explicitly seeds its new session.

The original private dataset and notebook version-2 outputs are attached as Kaggle inputs. No local dataset or weight upload is needed. Original results remain under `../outputs/download/v2/`; the teacher's six-class baseline is untouched. Checksum verification is concurrent and image bytes are cached in RAM to avoid repeating remote small-file reads during training.

The separate private continuation notebook is:

<https://www.kaggle.com/code/joanbalibrearull/cellscope-bloodmnist-continue-40>

One submission is permitted. The source run must be complete, and the submission attempt is recorded before the network request to prevent accidental duplicate jobs.

After submission, existing root commands automatically monitor the continuation:

```sh
../.venv/bin/python manage.py sync
../.venv/bin/python manage.py watch
../.venv/bin/python manage.py logs
```

Run those from the classification subproject. `watch` reads status and downloads all saved output when terminal. `logs` reads actual epoch records when Kaggle's stream is available; stream interruptions do not stop training. Refreshing the dashboard never launches or restarts training. Saved history must contain each epoch 1–40 exactly once before completion is published.

The continuation stores checkpoints after the scheduler update, while preserving all original 1–33 history rows. On completion, it updates the existing dashboard with all 40 measured epochs and performs local CPU inference on the downloaded selected model.

The test set was already evaluated for the original validation-selected checkpoint. If the same weights remain selected at epoch 40, existing predictions/reports are reused without another test inference pass. If validation selects different weights, the new checkpoint receives one test pass and the report explicitly records **two test evaluations across versions**; it must not be described as a fresh independent holdout. The fixed continuation budget is user requested and model selection continues to use validation only.

Five focused checks in `test_resume.py` verify real checkpoint parameter/moment restoration, the missing scheduler step, protection against advancing it twice, complete history, weight identity, and equivalent cached image decoding. Temporary fixture metrics never populate production reports.
