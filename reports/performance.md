# Performance report: unavailable

No performance metrics can be reported because Kaggle supplied a CPU runtime for every GPU-requested attempt. The training script deliberately stopped before training when CUDA was unavailable.

No fine-tuned model was produced, and no validation or test predictions were generated. All 126 test images remain unused for model selection or evaluation.

See the [execution report](execution.md) for the observed failures and required next step, and the [training protocol](training.md) for the planned evaluation. After successful GPU training, `manage.py sync` replaces this report with measured aggregate and per-class precision, recall, mAP50, mAP50–95 and timing results.
