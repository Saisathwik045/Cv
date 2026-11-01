# V11 execution and artifact audit

All six fits completed. See the separately generated prespecified paired comparison; this artifact audit does not perform selection.

Each published metric is recomputed from checksum/source/nonce-verified outer prediction files. Patient IDs, labels and containment groups match the frozen training-only outer assignments; mean-eye aggregation is reconstructed from saved eye probabilities. Best/stop state is replayed from inner history only. Patient-level artifacts stay private.

| Run | Status | Epochs / best | Outer macro AUROC | Macro AP | Macro F1 | Micro F1 | TP / FP / FN |
|---|---|---:|---:|---:|---:|---:|---:|
| fold0_ratio_bce | complete | 12 / 6 | 0.83453 | 0.57172 | 0.52226 | 0.55904 | 535 / 473 / 371 |
| fold0_sqrt_ratio_bce | complete | 9 / 3 | 0.84159 | 0.59376 | 0.50117 | 0.53726 | 465 / 360 / 441 |
| fold1_ratio_bce | complete | 12 / 6 | 0.83224 | 0.55951 | 0.52906 | 0.54527 | 545 / 576 / 333 |
| fold1_sqrt_ratio_bce | complete | 8 / 2 | 0.83930 | 0.57107 | 0.52249 | 0.49546 | 409 / 364 / 469 |
| fold2_ratio_bce | complete | 15 / 9 | 0.83675 | 0.58401 | 0.56570 | 0.53326 | 465 / 391 / 423 |
| fold2_sqrt_ratio_bce | complete | 10 / 4 | 0.82557 | 0.58566 | 0.51712 | 0.49938 | 404 / 326 / 484 |

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
| fold0_sqrt_ratio_bce | N | 263 / 496 | 0.7582 | 0.5838 | 0.5247 | 0.8479 | 0.6483 | 223 | 202 | 40 |
| fold0_sqrt_ratio_bce | D | 245 / 514 | 0.7755 | 0.6719 | 0.7750 | 0.3796 | 0.5096 | 93 | 27 | 152 |
| fold0_sqrt_ratio_bce | G | 46 / 713 | 0.9241 | 0.6440 | 0.6111 | 0.4783 | 0.5366 | 22 | 14 | 24 |
| fold0_sqrt_ratio_bce | C | 45 / 714 | 0.9841 | 0.9005 | 0.8684 | 0.7333 | 0.7952 | 33 | 5 | 12 |
| fold0_sqrt_ratio_bce | A | 35 / 724 | 0.8953 | 0.5226 | 0.7778 | 0.2000 | 0.3182 | 7 | 2 | 28 |
| fold0_sqrt_ratio_bce | H | 23 / 736 | 0.7732 | 0.1648 | 0.2500 | 0.0870 | 0.1290 | 2 | 6 | 21 |
| fold0_sqrt_ratio_bce | M | 37 / 722 | 0.9921 | 0.8587 | 0.7436 | 0.7838 | 0.7632 | 29 | 10 | 8 |
| fold0_sqrt_ratio_bce | O | 212 / 547 | 0.6302 | 0.4039 | 0.3733 | 0.2642 | 0.3094 | 56 | 94 | 156 |
| fold1_ratio_bce | N | 231 / 527 | 0.7618 | 0.5590 | 0.5099 | 0.6710 | 0.5794 | 155 | 149 | 76 |
| fold1_ratio_bce | D | 244 / 514 | 0.7682 | 0.6393 | 0.5251 | 0.6434 | 0.5783 | 157 | 142 | 87 |
| fold1_ratio_bce | G | 47 / 711 | 0.9284 | 0.4633 | 0.4737 | 0.3830 | 0.4235 | 18 | 20 | 29 |
| fold1_ratio_bce | C | 48 / 710 | 0.9889 | 0.8882 | 0.8200 | 0.8542 | 0.8367 | 41 | 9 | 7 |
| fold1_ratio_bce | A | 36 / 722 | 0.8855 | 0.4926 | 0.7500 | 0.2500 | 0.3750 | 9 | 3 | 27 |
| fold1_ratio_bce | H | 22 / 736 | 0.7052 | 0.2320 | 0.5000 | 0.2273 | 0.3125 | 5 | 5 | 17 |
| fold1_ratio_bce | M | 38 / 720 | 0.9878 | 0.8150 | 0.5211 | 0.9737 | 0.6789 | 37 | 34 | 1 |
| fold1_ratio_bce | O | 212 / 546 | 0.6321 | 0.3866 | 0.3650 | 0.5802 | 0.4481 | 123 | 214 | 89 |
| fold1_sqrt_ratio_bce | N | 231 / 527 | 0.7617 | 0.5602 | 0.6090 | 0.3506 | 0.4451 | 81 | 52 | 150 |
| fold1_sqrt_ratio_bce | D | 244 / 514 | 0.7509 | 0.6211 | 0.5138 | 0.6107 | 0.5581 | 149 | 141 | 95 |
| fold1_sqrt_ratio_bce | G | 47 / 711 | 0.9423 | 0.5062 | 0.4156 | 0.6809 | 0.5161 | 32 | 45 | 15 |
| fold1_sqrt_ratio_bce | C | 48 / 710 | 0.9793 | 0.8550 | 0.8163 | 0.8333 | 0.8247 | 40 | 9 | 8 |
| fold1_sqrt_ratio_bce | A | 36 / 722 | 0.9136 | 0.5587 | 0.8000 | 0.3333 | 0.4706 | 12 | 3 | 24 |
| fold1_sqrt_ratio_bce | H | 22 / 736 | 0.7425 | 0.2345 | 0.2778 | 0.2273 | 0.2500 | 5 | 13 | 17 |
| fold1_sqrt_ratio_bce | M | 38 / 720 | 0.9921 | 0.8496 | 0.7143 | 0.9211 | 0.8046 | 35 | 14 | 3 |
| fold1_sqrt_ratio_bce | O | 212 / 546 | 0.6321 | 0.3832 | 0.3873 | 0.2594 | 0.3107 | 55 | 87 | 157 |
| fold2_ratio_bce | N | 244 / 514 | 0.7254 | 0.5110 | 0.4955 | 0.6803 | 0.5734 | 166 | 169 | 78 |
| fold2_ratio_bce | D | 244 / 514 | 0.7018 | 0.5828 | 0.6353 | 0.4426 | 0.5217 | 108 | 62 | 136 |
| fold2_ratio_bce | G | 47 / 711 | 0.9395 | 0.6278 | 0.6279 | 0.5745 | 0.6000 | 27 | 16 | 20 |
| fold2_ratio_bce | C | 45 / 713 | 0.9609 | 0.8193 | 0.7400 | 0.8222 | 0.7789 | 37 | 13 | 8 |
| fold2_ratio_bce | A | 36 / 722 | 0.9184 | 0.6084 | 0.6000 | 0.5833 | 0.5915 | 21 | 14 | 15 |
| fold2_ratio_bce | H | 22 / 736 | 0.8300 | 0.1536 | 0.3000 | 0.1364 | 0.1875 | 3 | 7 | 19 |
| fold2_ratio_bce | M | 38 / 720 | 0.9979 | 0.9646 | 0.9211 | 0.9211 | 0.9211 | 35 | 3 | 3 |
| fold2_ratio_bce | O | 212 / 546 | 0.6202 | 0.4046 | 0.3886 | 0.3208 | 0.3514 | 68 | 107 | 144 |
| fold2_sqrt_ratio_bce | N | 244 / 514 | 0.7328 | 0.5150 | 0.5000 | 0.3852 | 0.4352 | 94 | 94 | 150 |
| fold2_sqrt_ratio_bce | D | 244 / 514 | 0.7495 | 0.6451 | 0.5918 | 0.5943 | 0.5930 | 145 | 100 | 99 |
| fold2_sqrt_ratio_bce | G | 47 / 711 | 0.9302 | 0.5834 | 0.6000 | 0.5106 | 0.5517 | 24 | 16 | 23 |
| fold2_sqrt_ratio_bce | C | 45 / 713 | 0.9672 | 0.7506 | 0.6842 | 0.8667 | 0.7647 | 39 | 18 | 6 |
| fold2_sqrt_ratio_bce | A | 36 / 722 | 0.8912 | 0.6279 | 0.8125 | 0.3611 | 0.5000 | 13 | 3 | 23 |
| fold2_sqrt_ratio_bce | H | 22 / 736 | 0.7233 | 0.2311 | 0.5000 | 0.0909 | 0.1538 | 2 | 2 | 20 |
| fold2_sqrt_ratio_bce | M | 38 / 720 | 0.9974 | 0.9483 | 0.9032 | 0.7368 | 0.8116 | 28 | 3 | 10 |
| fold2_sqrt_ratio_bce | O | 212 / 546 | 0.6131 | 0.3839 | 0.3960 | 0.2783 | 0.3269 | 59 | 90 | 153 |

GPU-quota totals come from saved API counters, not worker times. Per-session quota snapshots and receipts are private, with requested accelerator, timeout and source/checkpoint/output hashes preserved. The final global counter delta is authoritative; account intervals can include delayed billing.

No full-data refit, old calibration temperature, threshold tuning, acceptance policy or demo promotion occurs. Same-source data, earlier adaptive choices, sparse H and one seed still limit independence. A fresh suitable independent cohort is required for new performance claims. REVIEW REQUIRED remains unchanged.

[Machine-readable audit and hashes](execution-audit.json) · [Per-class CSV](executed-per-class.csv) · [Prospective protocol](../../docs/NESTED_LOSS_V11.md)
