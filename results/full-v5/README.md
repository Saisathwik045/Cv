# Validation development comparison

Generated from checksum-verified best-checkpoint prediction files. Patient probabilities are the mean of both eyes; threshold p >= 0.5. This is development evidence, not final evaluation.

| Backbone | State | Epochs / best | Patient macro AUROC | Macro AP | Macro F1 | Micro F1 |
|---|---|---:|---:|---:|---:|---:|
| efficientnet_b0 | complete | 5 / 2 | 0.7876 | 0.4686 | 0.4055 | 0.4230 |
| resnet50 | complete | 5 / 2 | 0.7784 | 0.4632 | 0.3708 | 0.3974 |
| densenet121 | complete | 6 / 3 | 0.8119 | 0.5053 | 0.4607 | 0.4773 |
| convnext_tiny | running | — | — | — | — | — |
| efficientnet_b2 | queued | — | — | — | — | — |

Selected backbone: pending completion/defined rankings for all five.

| Label | Validation positive / negative patients |
|---|---:|
| N | 175 / 350 |
| D | 169 / 356 |
| G | 32 / 493 |
| C | 32 / 493 |
| A | 25 / 500 |
| H | 16 / 509 |
| M | 26 / 499 |
| O | 147 / 378 |

H has only 16 positive validation patients; class-specific estimates are unstable. No calibration or acceptance-risk claims.

[Per-class and image/patient metrics](comparison.json) · [CSV](comparison.csv) · [AUROC plot](validation-auroc.png) · [Loss curves](validation-loss.png)
