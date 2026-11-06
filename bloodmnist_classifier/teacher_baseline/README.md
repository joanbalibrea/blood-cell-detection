# Teacher checkpoint: separate six-class baseline

The supplied checkpoint was found at [the existing model location](../models/resnet18-bloodmnist-wbc.pt). It is read in place; neither the checkpoint nor the sibling [BloodMNIST dataset](../../../bloodmnist/) is copied or changed.

This baseline has its own evaluation code and [report directory](../reports/generated/teacher_baseline/). It does not write the eight-class model's checkpoints, dashboard data, training source, outputs or Kaggle run state. The ongoing eight-class Kaggle job is left running.

The checkpoint explicitly declares ResNet18, six outputs in this order: basophil, eosinophil, immature granulocytes, lymphocyte, monocyte, neutrophil. Input size is 128; normalization is the supplied ImageNet mean/std. Platelets and erythroblasts cannot be predicted and are excluded from the six-class test subset.

The checkpoint does **not** specify a crop/interpolation recipe. The evaluation uses direct 128 × 128 bilinear resizing with antialiasing as a stated assumption. It must not be described as an exact reproduction of the teacher's workflow without the missing preprocessing information. The teacher's manifest hash and training provenance are stored as supplied metadata; their equivalence to local data/splits is unverified.

Run once from the classification subproject:

```sh
../.venv/bin/python teacher_baseline/evaluate.py
```

CPU inference uses two threads. Optional MPS: add `--device mps`. The script validates architecture/class metadata and loads weights strictly with `weights_only=True`. It verifies each evaluated image against the existing local manifest. It first evaluates validation without changing weights or tuning preprocessing, then evaluates the original six-class test subset once. It refuses an implicit repeated evaluation in the same report directory.

The primary six-class test subset has 2,640 images. A secondary sensitivity analysis removes known conflicting-label test images from the same saved probabilities; it does not perform another inference pass. Validation uses the existing non-overlap loader exclusions. No test image is removed from the eight-class experiment.

Outputs include JSON/CSV, every test probability, per-class precision/recall/F1, accuracy, macro metrics, weighted F1, one-vs-rest AUROC, count/normalized confusion matrices, selected correct/incorrect examples and measured local inference time. The standalone HTML report can be read directly or through the existing classifier server at:

<http://127.0.0.1:8766/reports/generated/teacher_baseline/index.html>

The existing server already permits this new report path; no eight-class dashboard/server files need modification. Start it as documented in the parent [README](../README.md) if needed.

Six-class closed-set metrics cannot be compared directly with the eight-class overall score. They differ in possible output labels, test denominator, architecture, provenance and preprocessing. A fair future comparison must explicitly define a common evaluation population and how eight-class predictions outside the six shared categories are counted.
