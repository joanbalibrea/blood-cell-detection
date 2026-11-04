# Dataset audit

| Split | Images | WBC | RBC | Platelets |
|---|---:|---:|---:|---:|
| train | 882 | 908 | 11220 | 382 |
| val | 252 | 257 | 3383 | 112 |
| test | 126 | 133 | 1699 | 49 |

Total: 1260 images, 18143 objects.
Invalid/missing labels: 0.
Exact duplicates across splits: 0.

No patient identifiers are available, so patient-level independence cannot be verified.
The three labels identify broad cell types; WBC subtype classification is not supported.
PNG color types: 2 = RGB, 6 = RGBA. Training inputs are converted to RGB without changing geometry.
Original splits are preserved; the test split is reserved for final evaluation.
