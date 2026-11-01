# V11 execution and artifact audit

**Incomplete matrix: no winner or overall paired conclusion is declared.**

Each published metric is recomputed from checksum/source/nonce-verified outer prediction files. Patient IDs, labels and containment groups match the frozen training-only outer assignments; mean-eye aggregation is reconstructed from saved eye probabilities. Best/stop state is replayed from inner history only. Patient-level artifacts stay private.

| Run | Status | Epochs / best | Outer macro AUROC | Macro AP | Macro F1 | Micro F1 | TP / FP / FN |
|---|---|---:|---:|---:|---:|---:|---:|
| fold0_ratio_bce | complete | 12 / 6 | 0.83453 | 0.57172 | 0.52226 | 0.55904 | 535 / 473 / 371 |
| fold0_sqrt_ratio_bce | running | — | — | — | — | — | — |
| fold1_ratio_bce | queued | — | — | — | — | — | — |
| fold1_sqrt_ratio_bce | queued | — | — | — | — | — | — |
| fold2_ratio_bce | queued | — | — | — | — | — | — |
| fold2_sqrt_ratio_bce | queued | — | — | — | — | — | — |

## All-eight completed-run counts

| Run | Class | + / − | AUROC | AP | Precision | Recall | F1 | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fold0_ratio_bce | N | 263 / 496 | 0.7691 | 0.5966 | 0.5983 | 0.5323 | 0.5634 | 140 | 94 | 123 |
| fold0_ratio_bce | D | 245 / 514 | 0.7806 | 0.6816 | 0.5134 | 0.7837 | 0.6204 | 192 | 182 | 53 |
| fold0_ratio_bce | G | 46 / 713 | 0.9147 | 0.6095 | 0.6667 | 0.3913 | 0.4932 | 18 | 9 | 28 |
| fold0_ratio_bce | C | 45 / 714 | 0.9881 | 0.8655 | 0.7321 | 0.9111 | 0.8119 | 41 | 15 | 4 |
| fold0_ratio_bce | A | 35 / 724 | 0.8922 | 0.4665 | 0.6111 | 0.3143 | 0.4151 | 11 | 7 | 24 |
| fold0_ratio_bce | H | 23 / 736 | 0.6864 | 0.1194 | 0.1111 | 0.0435 | 0.0625 | 1 | 8 | 22 |
| fold0_ratio_bce | M | 37 / 722 | 0.9856 | 0.8138 | 0.7317 | 0.8108 | 0.7692 | 30 | 11 | 7 |
| fold0_ratio_bce | O | 212 / 547 | 0.6596 | 0.4209 | 0.4096 | 0.4811 | 0.4425 | 102 | 147 | 110 |

GPU-quota totals come from saved API counters, not worker times. Per-session quota snapshots and receipts are private, with requested accelerator, timeout and source/checkpoint/output hashes preserved. The final global counter delta is authoritative; account intervals can include delayed billing.

No full-data refit, old calibration temperature, threshold tuning, acceptance policy or demo promotion occurs. Same-source data, earlier adaptive choices, sparse H and one seed still limit independence. A fresh suitable independent cohort is required for new performance claims. REVIEW REQUIRED remains unchanged.

[Machine-readable audit and hashes](execution-audit.json) · [Per-class CSV](executed-per-class.csv) · [Prospective protocol](../../docs/NESTED_LOSS_V11.md)
