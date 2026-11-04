# Execution status: GPU provisioning blocked

Training has **not completed**. No fine-tuned weights or measured model accuracy are available.

The private Kaggle dataset and training kernel were created successfully. Dataset audit and local pipeline tests passed. The notebook can be accessed by its owner at <https://www.kaggle.com/code/joanbalibrearull/txl-pbc-yolo26-training>.

## Observed attempts

| Version | Outcome |
|---|---|
| 1 | Package installation failed because the Kaggle session could not resolve PyPI. No training ran. |
| 2 | Offline package installation succeeded. CUDA was unavailable. No training ran. |
| 3 | Explicit T4 request; diagnostics showed CPU-only PyTorch, no CUDA build, no GPUs and no `nvidia-smi`. |
| 4 | Explicit T4 plus official GPU-container digest. Kaggle again used the CPU container. No training ran. |

## Evidence

- Requested accelerator: `NvidiaTeslaT4`, `enable_gpu: true`.
- Requested container: `gcr.io/kaggle-gpu-images/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01`.
- Returned container: `gcr.io/kaggle-images/python@sha256:17b2fc7f27c25c6c2fd4230ec8e66097ab2b89fc88e1e4d6f338b72e1ee0bc3e`.
- Runtime PyTorch: `2.11.0+cpu`; CUDA build: `null`; GPU count: `0`; `nvidia-smi` unavailable.
- Kaggle quota API reported 6 hours allowed, zero used/reserved and `hasEverRun: false`.
- [Runtime diagnostics](startup.json) record the actual session values.

This establishes a Kaggle accelerator/runtime provisioning problem. It does not establish whether the cause is account verification, accelerator eligibility, availability, or API behavior. The browser available to this session is not signed in, so the private notebook's accelerator controls could not be inspected.

## Required next step

While signed in to the notebook owner's Kaggle account, open the notebook editor and check **Settings → Accelerator**. Enable a T4 GPU, and complete any verification Kaggle requests. Confirm that a session reports CUDA available. If the GPU choice is disabled, its message identifies the remaining account restriction.

Then retry the prepared offline training entrypoint. Local commands and dashboard instructions are in the [README](../README.md). Do not upload datasets again merely to resume. Use `manage.py launch` only to start a new training version; `manage.py watch` and `stream_logs.py` monitor it.

The Git repository is initialized on `main`. No commit, GitHub remote, or push was created.
