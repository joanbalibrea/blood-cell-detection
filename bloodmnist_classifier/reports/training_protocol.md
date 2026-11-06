# Classifier training and interpretation protocol

## Study question

Can an ImageNet-pretrained EfficientNet-B0 classify low-resolution BloodMNIST images into the eight supplied categories? This experiment evaluates an independent image classifier. Detection and later use on YOLO crops are separate experiments.

## Data provenance and integrity

Use the supplied local dataset at the workspace root, preserving original split membership and all source files. The [dataset audit](generated/dataset_report.md) gives the verified counts and representative examples. The [manifest](generated/manifest.csv) records relative path, label, dimensions, format, and byte/pixel hashes for every image. Kaggle verifies the inventory and each byte hash before training.

Nine exact groups overlap splits. The [selected loader policy](generated/selected_data_policy.json) excludes eight overlapping training images and one overlapping validation image; all 3,421 test images remain. Effective training/validation counts are 11,951 and 1,711. There is no random repartitioning. The policy uses image identity to prevent leakage, not model test results. All classes remain represented. One exact image has conflicting eosinophil/neutrophil labels; the original labels are retained, and the overlapping training entry is excluded.

Patient independence, near-duplicate independence, label adjudication and the clinical sampling frame are unavailable. Images may show surrounding cells; a category label does not guarantee that only one biological object is visible. The platelet example contains multiple platelet-like structures. The present task is image classification under the supplied labels.

## Configuration

| Setting | Value |
|---|---|
| Architecture | EfficientNet-B0, eight-class linear head |
| Initialization | Official ImageNet-1K V1 weights, hash verified |
| Trainable parameters | 4,017,796 |
| Input | RGB, 224 × 224 |
| Loss | Unweighted cross-entropy |
| Optimizer | AdamW |
| Initial learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Batch size | 64 |
| Epoch limit | 40 |
| Early stopping | Patience 8, minimum meaningful validation macro F1 gain 0.0001 |
| Scheduler | ReduceLROnPlateau on validation macro F1, factor 0.5, patience 3 |
| Checkpoint selection | Highest observed validation macro F1 |
| Seed | 42, Python/NumPy/PyTorch/CUDA/loader workers |
| Precision | Training: CUDA FP16 autocast + gradient scaling; validation/test: FP32 |
| Requested accelerator | Kaggle T4 |
| Device use | CUDA GPU 0; additional allocated GPUs are not used |

Training augmentation: random resized crop (scale 0.85–1.0), random horizontal/vertical flips, ±20° rotation and mild color jitter. Evaluation/inference: bicubic resize to 256, center crop 224, tensor conversion, ImageNet mean/std normalization. Enlarging a 64-pixel image adds no biological information. Mild staining augmentation is a hypothesis tested through validation performance, not a guarantee of domain robustness.

Unweighted loss keeps the empirical image distribution. Macro F1 gives every class equal weight for checkpoint selection, while per-class precision/recall/support reveal minority-class behavior. Future changes such as class weighting must use validation evidence and a separately documented experiment; the test set must not drive tuning.

## Monitoring and interpretation

Training accuracy/loss measure augmented training batches and are not directly identical to evaluation-mode accuracy/loss. Every epoch evaluates the deterministic validation split. Loss, accuracy, validation macro F1, learning rate, GPU and elapsed time are stored as CSV/JSON and emitted as structured log records.

The dashboard can receive actual epoch log records, if Kaggle exposes the stream, or final saved outputs after termination. Status polling alone is not live metric monitoring. A decreasing training loss with increasing validation loss is flagged as a potential generalization warning; inspect the trend over multiple epochs and validation macro F1 before interpreting it as overfitting.

## Final evaluation

Only after checkpoint selection is complete, perform one FP32 inference pass on the complete original test split. Compute accuracy; macro precision/recall/F1; weighted F1; per-class precision/recall/F1/support; count and true-row-normalized confusion matrices; and macro one-vs-rest AUROC if all required positive/negative support exists. Tied probabilities are handled together in the ROC calculation. Undefined precision/recall/F1 is assigned zero; unsupported AUROC is unavailable rather than invented.

Pay special attention to immature granulocyte/neutrophil/eosinophil confusion. Save all image predictions and class probabilities so results can be audited. Display one correct example and up to two high-confidence errors per true class, when available; these are selected diagnostic examples, not a representative random sample. Softmax scores are uncalibrated.

Measure batched forward time separately from total image loading/transfers/probability conversion. Exclude warmup from timing and record the device. These measurements do not establish single-image Mac latency.

## Deployment boundary

The checkpoint stores CPU-readable weights, architecture, class order, normalization, input size and validation-selected epoch. The local inference module defaults to CPU, optionally supports MPS, and requires no network access. It accepts an individual cell image and returns a class, confidence and eight probabilities. Trained-weight inference must be verified only after a real checkpoint is downloaded. Clinical use, stain/domain adaptation, donor-level validation and detector/classifier integration are outside this experiment.
