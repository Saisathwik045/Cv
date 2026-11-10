# V13 matched resolution findings

All six fresh-ImageNet ConvNeXt-Tiny ratio-BCE fits completed. The unchanged training-only containment folds, common microbatch8/accumulation2, inner-only checkpoint selection, raw mean-eye sigmoid predictor and fixed p≥.5 define this comparison. V11 checkpoints were not used as references. No historical final, calibration, threshold fitting, full-data refit or demo change.

| Arm | Mean outer macro AUROC | Mean macro AP | Mean macro F1 | Mean micro F1 |
|---|---:|---:|---:|---:|
| res224 | 0.83273 | 0.57279 | 0.53758 | 0.53534 |
| res320 | 0.85108 | 0.59412 | 0.57156 | 0.57509 |

| Paired mean difference (320−224) | Estimate | Conditional 95% interval | Defined / total |
|---|---:|---:|---:|
| macro_auroc | +0.01835 | [+0.01036, +0.02655] | 2000 / 2000 |
| macro_auprc_average_precision | +0.02133 | [+0.00829, +0.03713] | 2000 / 2000 |
| macro_f1 | +0.03398 | [+0.01370, +0.05323] | 2000 / 2000 |
| H_recall | +0.07049 | [-0.01515, +0.15219] | 2000 / 2000 |
| H_f1 | -0.01432 | [-0.09749, +0.06137] | 2000 / 2000 |

The conditional paired primary interval favors320 on this development cohort; this is not independent confirmation. Any macro F1 increase must be considered alongside all-class false-positive/false-negative tradeoffs.
The prespecified numerical rule nominates res320 for future study only. There is no global fold-checkpoint selection, deployment decision or automatic refit.

Intervals use2000 paired component-cluster bootstrap replicates, fixed seed2026, sampling within each outer fold with the same groups for both arms. They condition on fitted models and omit training/checkpoint-selection variability. Previous adaptive choices, same-source data, three folds and one seed limit independence. Outer sample separation inside each fit does not erase prior development exposure.

[All-eight per-class metrics, H/O tradeoffs, plots and limitations](DISCUSSION.md) · [Verified execution](EXECUTION.md) · [Settled accounting and artifact hashes](execution-provenance.json)
