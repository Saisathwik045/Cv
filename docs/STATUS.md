# Phase 5 complete — five-backbone validation comparison

PR #11 was already merged (`3ac4032`) and its final CI passed. The fixed v4
cohort, architecture, five backbones and patient aggregation remain unchanged.
[Historical phase 4 evidence](history/STATUS-through-phase4.md) is preserved.
No duplicate investigation, new splits, GPU smoke, paid compute, calibration
fitting, risk-policy fitting or final-evaluation access occurred in this phase.

## Frozen experiment and evidence

Frozen v4 manifest SHA256:
`c094f372aa8e9e2b15d1483c8f0f9cab58f3e3beabc11e938fd281f205c7af57`.
Train/validation: 2,275/525 patients and 4,550/1,050 images. Readiness, policy,
role and pilot checksums were reused; historical manifests were preserved.
[Reuse receipt](../results/full-v5/reused-evidence.json) records the hashes.

The existing CPU input job was recovered, not replaced:
`saisathwikregula/trustretina-full-v5-inputs-4f23af6d/1` (510 worker seconds).
Its source, nonce, private/CPU flags and every output hash passed. Input-index SHA:
`a45f0dd9b1e9d49b89014ee3b3e3104f451f3fa9806c824d2ba07cdf17509888`.
GPU jobs mounted only this train/validation cache and official ImageNet weights;
original data, calibration and final role files were absent.

[Common config](../configs/full-backbones-v5.toml) SHA256:
`99e309ac619f73bec615ed32b08402a951e764f3a22187ecd5af0cb7bda59b89`.
All five started from ImageNet V1, never the pilot checkpoint. Seed 2026, common
224 letterbox/ImageNet normalization, FP32 AdamW, batch 16, training-only weighted
BCE, maximum ten epochs and fixed patience three/delta 0.0001. The same source
revision `55d266a` and source-tree hash govern every worker. Actual runtime package
versions/freezes are retained with the private outputs. See [protocol](FULL_EXPERIMENT_V5.md).

## Actual completed runs

| Backbone | Epochs / best | Patient macro AUROC | Pinned private job/version |
|---|---:|---:|---|
| efficientnet_b0 | 5 / 2 | 0.7876 | `saisathwikregula/trustretina-full-v5-efficientnet-b0-6c3a859e/1` |
| resnet50 | 5 / 2 | 0.7784 | `saisathwikregula/trustretina-full-v5-resnet50-618fc487/1` |
| densenet121 | 6 / 3 | 0.8119 | `saisathwikregula/trustretina-full-v5-densenet121-a0a7f6e8/1` |
| convnext_tiny | 5 / 2 | 0.8288 | `saisathwikregula/trustretina-full-v5-convnext-tiny-c349c706/1` |
| efficientnet_b2 | 5 / 2 | 0.8048 | `saisathwikregula/trustretina-full-v5-efficientnet-b2-1c489932/1` |

Selected backbone: **convnext_tiny**, using maximum validation patient macro AUROC
of each model's minimum patient-validation-weighted-BCE checkpoint. Exact ties
retain the fixed backbone order. No other metric/epoch or final result selected it.
Patient probabilities are the mean of both eye sigmoid outputs; initial threshold
is 0.5. These are development selection results, not final performance.

[Generated comparison, per-class metrics and plots](../results/full-v5/README.md)
come from actual checksum-verified NPZ predictions. All eight ranking metrics are
defined; validation H has only 16 positive patients, so class estimates remain
unstable. Labels are patient-derived, not independently verified eye diagnoses.

## Artifacts, runtime and recovery

Best/last checkpoints, history/timing, dependency freezes, image/patient predictions
and logs were retrieved and verified for every run. Source/privacy/resources,
nonce, source revision, report, all artifact hashes and checkpoint/progress binding
passed. Each best checkpoint was strictly reloaded remotely; epoch-boundary
resume state includes optimizer and Python/NumPy/Torch/CUDA/loader RNG. No retry
or resumed attempt was needed. Independent B0 and selected ConvNeXt best-checkpoint/report downloads into fresh
directories matched [B0 hashes](../results/full-v5/fresh-retrieval-proof.json) and
[selected-checkpoint hashes](../results/full-v5/selected-fresh-retrieval-proof.json).

Observed quota counter increase: **1882.137 seconds (0.523 hours)**.
Summed run-function elapsed time: **1707.274 seconds**; it includes CPU/I/O
and excludes interpreter imports/provisioning. Accelerator kernel time is not
separately measured. Paid scaling was disabled. [Timing/quota receipt](../results/full-v5/gpu-time.json).

Large/sample-level outputs remain in pinned private Kaggle kernel-version storage
and ignored local bundles. The permitted [registry](../results/full-v5/run-registry.json)
records job/version IDs, source and artifact hashes, with no patient records.
Private index: `artifacts/runner/full-v5/PRIVATE_INDEX.md`. No monitor remains active.

Fresh-task recovery uses the configured account:

```sh
python scripts/full_experiment.py restore  # only when private registry is absent
python scripts/full_experiment.py retrieve
python scripts/compare_full_experiment.py
python scripts/full_experiment.py export
```

Use the existing Kaggle SDK environment for recovery and the locked
`requirements/full-v5-reports.lock` CPU environment for comparison. `advance --submit`
is a no-op once all five are complete; never initialize/rebuild these runs.

Local checks: 116 tests passed, nine optional/environment skips; all five strict
ImageNet factory loads passed. PR #12 implementation CI passed on Python 3.11/3.12;
final evidence commit checks are tracked in that PR.

## Next executable phase

Freeze the selected best checkpoint below and proceed to reliability and shared-
temperature calibration experiments under the frozen v4 protocol. Confidence,
MC Dropout, quality, calibration, triage and Grad-CAM remain in the architecture.
No calibration or acceptance-risk parameters were fitted here. Rare calibration
class support cannot justify class-specific fitting, unsupported policy fitting
or clinical guarantees. Final evaluation remains untouched.

Selected best checkpoint SHA256:
`6871e49772543f3463f9d30dd6c758257022463bf562e21342fd154d7aa0eaf1`.
