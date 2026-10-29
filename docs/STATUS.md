# Phase 7 complete — held-out research evaluation; frozen demo packaging next

PR #13 was already merged at `c4d563a`; its final CI passed. Work continued from
latest main. [Phase 6 history](history/STATUS-through-phase6.md) preserves the
candidate selection/calibration evidence. No app rebuild, training, parameter
refit, split change, duplicate review, seed search or paid compute occurred.

## Protocol and immutable reuse

[Final protocol](FINAL_EVALUATION_V7.md) and [configuration](../configs/final-evaluation-v7.toml)
were committed at `fed7e5a` **before final labels/images were opened**. An append-only
exposure ledger records the authorized first manifest read; its exact text/hash
is included in the permitted run-registry snapshot for recovery.

Candidate contract SHA256:
`a80dce8f27f12645539a0312a3ed96af0b5d10b5d7e1cd522af53b9347409c61`.
Selected ConvNeXt-Tiny epoch-two checkpoint SHA256:
`6871e49772543f3463f9d30dd6c758257022463bf562e21342fd154d7aa0eaf1`.
Frozen v4 root manifest SHA256:
`c094f372aa8e9e2b15d1483c8f0f9cab58f3e3beabc11e938fd281f205c7af57`.
Final role, historical role separation, selected checkpoint, saved temperature,
normalizer and development-cutoff artifacts all passed checksum verification.
Raw inputs/labels/manifests remain unchanged. **All 350 final patients / 700 eyes /
346 containment components** were included. The holdout has now been evaluated:
do not use it for future parameter, threshold or strategy selection.

## Actual held-out results

Patient-equal primary raw macro AUROC: **0.8580 [0.8366, 0.8798]**.
Macro F1: **0.4774 [0.4335, 0.5150]**.
Micro F1: **0.5124 [0.4847, 0.5420]**.
All brackets are fixed-seed 95% component-cluster percentile intervals, conditional
on the frozen candidate. Exact-match accuracy is **16.0%**; labelwise accuracy
is **79.61%**; Hamming loss is **20.39%**. Full per-class raw/calibrated AUROC, AP,
precision/recall/F1/support, both weightings, valid replicate counts and clearly
defined accuracy metrics are in the [generated research report](../results/final-v7/REPORT.md).

The saved T=0.7226021421263825 and training ECDF normalizer were reused unchanged.
Calibration acts after the mean eye sigmoid probabilities. At p >= 0.5 all binary
decisions remain identical. Patient NLL **0.428021 → 0.419557**, Brier
**0.139747 → 0.137945**, macro ECE **0.187333 → 0.166800**.
Paired calibrated-minus-raw differences:
- NLL: **-0.008465 [-0.013116, -0.004062]**.
- Brier: **-0.001802 [-0.003052, -0.000733]**.
- Macro ECE: **-0.020534 [-0.022739, -0.015590]**.

Pooled losses improved on this internal holdout; classwise results were mixed.
**D and H NLL worsened**, with paired secondary intervals above zero. H has only
ten positive patients. No class-specific fitting or clinical guarantee follows.

| Fixed strategy | Raw Hamming AURC | Calibrated Hamming AURC |
|---|---:|---:|
| Confidence | 0.164433 | 0.164433 |
| + trained-head uncertainty | 0.164565 | 0.164581 |
| + technical quality | 0.191433 | 0.189610 |

Lower is better. MC differences versus confidence include zero; combined quality
worsened ranking. All six variants and all negative findings remain reported.
Saved-grid cutoffs came from validation and were applied unchanged; actual final
coverage differs. Full curves and grid subsets are **retrospective research**,
not a validated automatic acceptance policy. Accepted support/FN rates for all
eight labels and any-label-error risks are saved, including undefined subsets.
Deployment remains **REVIEW REQUIRED for every patient**; ACCEPT count is zero.

## Contribution, limitations and recovery

Results support the executable, reproducible research pipeline and modest pooled
calibration gains on one held-out cohort. They do **not** establish improved
reliability from adding head uncertainty/quality, clinical automatic triage,
complete epistemic uncertainty, clinical quality grading or lesion localization.
Patient-derived labels, sparse classes, one internal dataset, unknown residual
identity overlap, fixed-candidate bootstrap and non-simultaneous secondary intervals
limit generalization. No result changed the frozen pipeline.

All three private Kaggle stages completed once; no failed/replaced attempt or
corrected evaluation was needed. Sources/nonces, model-state invariance, exact
patient/eye/label/component alignment, 20 saved seeds (62026..62045) and all retrieved
artifact hashes passed. Private arrays and full score thresholds remain outside Git.

| Stage | Pinned private job/version | Run-function seconds |
|---|---|---:|
| analysis | `saisathwikregula/trustretina-phase7-analysis-dc337486/1` | 51.195 |
| inference | `saisathwikregula/trustretina-phase7-inference-885749c8/1` | 22.353 |
| inputs | `saisathwikregula/trustretina-phase7-inputs-5b25cfc4/1` | 270.066 |

Observed GPU quota increase: **68.087 seconds**, paid scaling disabled.
Run-function times exclude interpreter/provisioning overhead. Independent fresh-
directory retrieval of final predictions, intervals, configuration and frozen
parameters matched [recorded hashes](../results/final-v7/fresh-retrieval-proof.json).

Local Torch/report suite: **138 tests passed, nine optional skips**. Model-free
suite: **138 passed, 22 optional skips**. CI is tracked in the focused Phase 7 PR.
No monitoring process remains active. Private index: `artifacts/runner/phase7/PRIVATE_INDEX.md`.

Fresh-task artifact recovery using the existing account:

```sh
python scripts/final_evaluation.py restore  # only if private registry is absent
python scripts/final_evaluation.py retrieve
python scripts/publish_final_research.py
python scripts/final_evaluation.py export
```

Do not rerun initialization, completed inference/analysis, calibration or training.
The [registry](../results/final-v7/run-registry.json) records exact versions/receipts.
**Next phase: package the unchanged frozen inference pipeline into the demo app**,
with research labeling and REVIEW REQUIRED default. No new experiments or
parameter tuning against this now-used holdout are part of that phase.
