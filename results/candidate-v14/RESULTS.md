# V14 two-fit development results

Both frozen fits completed and outputs passed source/nonce/report/checksum verification. These are reused-validation development results, not independent confirmation. No model promotion, temperature/threshold fitting, historical final access or demo modification.

| Fit | Epochs / best | Macro AUROC | Macro AP | Macro F1 | Micro F1 |
|---|---:|---:|---:|---:|---:|
| 320 | 11 / 5 | 0.8596 | 0.5993 | 0.5371 | 0.5532 |
| 224 | 11 / 5 | 0.8480 | 0.5812 | 0.5332 | 0.5290 |

Raw patient probabilities are float64 mean left/right sigmoid probabilities; fixed p≥0.5. TRAIN-only ratio BCE; validation macro AUROC selects each checkpoint.

| Class | Resolution | Positive / negative | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N | 320 | 175 / 350 | 0.5116 | 0.8857 | 0.6485 | 155 | 148 | 20 |
| N | 224 | 175 / 350 | 0.4611 | 0.8800 | 0.6051 | 154 | 180 | 21 |
| D | 320 | 169 / 356 | 0.7561 | 0.3669 | 0.4940 | 62 | 20 | 107 |
| D | 224 | 169 / 356 | 0.7222 | 0.2308 | 0.3498 | 39 | 15 | 130 |
| G | 320 | 32 / 493 | 0.5926 | 0.5000 | 0.5424 | 16 | 11 | 16 |
| G | 224 | 32 / 493 | 0.4500 | 0.5625 | 0.5000 | 18 | 22 | 14 |
| C | 320 | 32 / 493 | 0.2385 | 0.9688 | 0.3827 | 31 | 99 | 1 |
| C | 224 | 32 / 493 | 0.4828 | 0.8750 | 0.6222 | 28 | 30 | 4 |
| A | 320 | 25 / 500 | 0.6875 | 0.4400 | 0.5366 | 11 | 5 | 14 |
| A | 224 | 25 / 500 | 0.5455 | 0.7200 | 0.6207 | 18 | 15 | 7 |
| H | 320 | 16 / 509 | 0.4000 | 0.2500 | 0.3077 | 4 | 6 | 12 |
| H | 224 | 16 / 509 | 0.1923 | 0.3125 | 0.2381 | 5 | 21 | 11 |
| M | 320 | 26 / 499 | 0.8800 | 0.8462 | 0.8627 | 22 | 3 | 4 |
| M | 224 | 26 / 499 | 0.8750 | 0.8077 | 0.8400 | 21 | 3 | 5 |
| O | 320 | 147 / 378 | 0.4413 | 0.6395 | 0.5222 | 94 | 119 | 53 |
| O | 224 | 147 / 378 | 0.4216 | 0.5850 | 0.4900 | 86 | 118 | 61 |

## Explicit class-specific limitations

- H false positives: 320=6, 224=21; false negatives: 320=12, 224=11.
- O false positives: 320=119, 224=118; false negatives: 320=53, 224=61.
- D missed positives: 320=107, 224=130, among 169 validation positives.
- A missed positives: 320=14, 224=7, among 25 validation positives.
- M missed positives: 320=4, 224=5, among 26 validation positives.

The descriptive320−224 differences are macro AUROC+0.01163, macro AP+0.01816, macro F1+0.00387 and micro F1+0.02419. No independent confirmation or significance claim follows.

Additional regressions must not be hidden:
- C:320 precision/recall/F1=0.2385/0.9688/0.3827, FP/FN=99/1;224=0.4828/0.8750/0.6222, FP/FN=30/4.
- G:320 precision/recall/F1=0.5926/0.5000/0.5424, FP/FN=11/16;224=0.4500/0.5625/0.5000, FP/FN=22/14.

| Resolution | Exact-match accuracy | Labelwise binary accuracy | Hamming loss |
|---|---:|---:|---:|
| 320 | 0.2990 | 0.8481 | 0.1519 |
| 224 | 0.3143 | 0.8436 | 0.1564 |

320 has lower exact-match accuracy despite its higher aggregate ranking/F1 scores. H detection remains weak (only4/16 positive patients detected at320), and C/A/G/H tradeoffs preclude a claim of uniform class improvement. Labelwise accuracy is not diagnostic accuracy. No independent-cohort endpoint or grouped interval is reported here; those remain unexecuted, separately gated research steps.

All-eight AUROC/AP and TN are in validation-per-class.csv. Sparse H support and reused validation limit interpretation. BRSET remains an independent access/compatibility blocker. The original demo remains REVIEW REQUIRED. Large checkpoints, aligned predictions, stochastic arrays and logs are retained in the pinned private job versions; hashes are recorded in execution-results.json.
