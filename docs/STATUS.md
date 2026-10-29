# Phase 5 — authorized full comparison started

PR #11 is already merged (`3ac4032`); its final CI passed. The historical review,
v4 release and pilot evidence are preserved in
[phase 4 status](history/STATUS-through-phase4.md). Architecture and five backbones
are unchanged. No duplicate investigation, split generation or GPU smoke was repeated.

## Verified inputs and fixed experiment

Frozen v4 manifest SHA256:
`c094f372aa8e9e2b15d1483c8f0f9cab58f3e3beabc11e938fd281f205c7af57`.
Readiness, containment policy, role checksums and successful pilot provenance were
verified and reused. Training/validation have 2,275/525 patients (4,550/1,050 eyes).
Calibration/final role contents were not opened during this phase.

The existing CPU cache job completed in 510 worker seconds and was recovered,
without replacement: `saisathwikregula/trustretina-full-v5-inputs-4f23af6d/1`.
Source, nonce, resource/privacy flags, report and every output checksum passed.
Input index SHA256:
`a45f0dd9b1e9d49b89014ee3b3e3104f451f3fa9806c824d2ba07cdf17509888`.
All five official ImageNet V1 files passed hash checks and strict factory loads.
Only train/validation cache and weights are attached to GPU jobs; original data,
calibration and final sets are not attached.

[Common configuration](../configs/full-backbones-v5.toml) SHA256:
`99e309ac619f73bec615ed32b08402a951e764f3a22187ecd5af0cb7bda59b89`.
[Full protocol](FULL_EXPERIMENT_V5.md): fresh ImageNet initialization, FP32 AdamW,
training-only BCE weights, common 224 preprocessing, batch 16, maximum ten epochs,
patient-validation-loss early stopping (patience three, delta 0.0001). Best
checkpoint minimizes that loss; backbone selection maximizes validation patient
macro AUROC, with all five required and fixed-order ties. Initial threshold 0.5.
No fitting of calibration or acceptance-risk policy occurs here.

## Current queue — last observed, not background monitoring

| Backbone | State | Pinned job/version |
|---|---|---|
| EfficientNet-B0 | Complete: 5 epochs | `saisathwikregula/trustretina-full-v5-efficientnet-b0-6c3a859e/1` |
| ResNet50 | Complete: 5 epochs | `saisathwikregula/trustretina-full-v5-resnet50-618fc487/1` |
| DenseNet121 | Complete: 6 epochs | `saisathwikregula/trustretina-full-v5-densenet121-a0a7f6e8/1` |
| ConvNeXt-Tiny | running | `saisathwikregula/trustretina-full-v5-convnext-tiny-c349c706/1` |
| EfficientNet-B2 | queued | — |

EfficientNet-B0 completed five epochs; best checkpoint is epoch two. Its best/last
checkpoints, aligned predictions, history and logs were retrieved and verified.
The [generated partial comparison](../results/full-v5/README.md) uses actual NPZ
predictions; no selected backbone is claimed until all five finish. Best/last checkpoints, predictions and histories are generated
by each worker and retrieved/checksummed after it finishes. Every completed epoch
writes a durable progress report for compatible resume. Ambiguous/failed submissions
block automatic retries. GPU jobs run sequentially on free Kaggle quota only.

At pre-submission inspection Kaggle reported 21,600 GPU seconds total, 210.693 used,
zero reserved and paid scaling disabled: approximately 5.94 GPU hours remained.
This is a quota observation, not full-training elapsed time. The completed B0 worker used 169.3 seconds (including CPU/I/O inside the GPU
session); the quota counter increased by 210.022 seconds. Exact accelerator kernel
time is not separately measured. Later runs remain in progress.

## Resume exactly

The private registry is `artifacts/runner/full-v5/registry.json`; receipts, logs,
checkpoints and sample-level predictions remain ignored. The permitted aggregate
[registry snapshot](../results/full-v5/run-registry.json) records pinned versions,
source hashes and receipts, with no patient records. Private details and checksums
are indexed at `artifacts/runner/full-v5/PRIVATE_INDEX.md`.

Use the configured Kaggle credentials without printing them. In this workspace:

```sh
KAGGLE_CONFIG_DIR=/workspace/shared/trustretina-kaggle /workspace/shared/trustretina-venv/bin/python scripts/full_experiment.py advance --submit
```

One call checks the current job; after completion it retrieves and verifies outputs
and may submit the next model. It returns while a job runs. Call at bounded intervals
of at least 60 seconds; no monitoring continues after this task ends. On a fresh
checkout, run `scripts/full_experiment.py restore` first **only if the private
registry is missing**. This restores receipts from the aggregate snapshot; the next
advance retrieves pinned remote artifacts. Never rerun init or cache preparation.

After completed models are available, install the CPU/report environment from
`requirements/full-v5-reports.lock` and use:
`python scripts/compare_full_experiment.py`, then `python scripts/full_experiment.py export`.
Comparison reads actual aligned NPZ predictions; it selects no winner until all five
complete. Keep generated permitted aggregate CSV/plots explicit when adding them
(the broad artifact ignore rules remain). Never add sample predictions/checkpoints.

Local checks: 116 tests passed, nine optional/environment skips; all five strict
ImageNet factory loads passed. PR #12 CI passed on both Python versions; later evidence updates remain tracked there.
Next executable step is advancing the existing queue. Reliability and shared-
temperature calibration experiments follow the completed validation comparison;
rare-class limitations remain, and final evaluation stays untouched.
