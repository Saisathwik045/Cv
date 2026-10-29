# Phase 6 complete — frozen research candidate; automatic triage unsupported

PR #12 was already merged at `5b059f2`; its final CI passed. Work continued from
latest main. [Phase 5 history](history/STATUS-through-phase5.md) retains all five
completed backbone runs. No backbone retraining, duplicate review, new splits,
GPU smoke, paid resources or final-evaluation access occurred.

## Fixed protocol and verified reuse

[Phase 6 protocol](RELIABILITY_V6.md) and [configuration](../configs/reliability-v6.toml)
were committed at `d837019` **before calibration-label access**. They fix predictors,
aggregation, seeds, twenty passes, quality normalization, the single-temperature
objective, fixed ablations/coverage grid, descriptive bootstrap and compute caps.
Frozen v4 manifest remains:
`c094f372aa8e9e2b15d1483c8f0f9cab58f3e3beabc11e938fd281f205c7af57`.
Selected ConvNeXt-Tiny epoch-two checkpoint remains:
`6871e49772543f3463f9d30dd6c758257022463bf562e21342fd154d7aa0eaf1`.
Frozen permitted role hashes, original readiness and selected training evidence
passed verification. Final role contents/images/labels were not opened.

## Actual components and findings

The existing trained `head.0` Dropout p=0.2 supports head-only MC inference.
Backbone features are cached once; stochastic depth and normalization stay eval.
Twenty fixed-seed passes aggregate both eyes within each pass. Their variability
is not complete backbone uncertainty or validated clinical uncertainty. Model
parameters/buffers were unchanged. All 1,050 deterministic validation image
predictions reproduced Phase 5 exactly (maximum absolute difference zero).

Quality features are foreground fraction, contrast and gradient energy on trimmed
canonical input. Training-only midrank ECDFs normalize them; patient quality is
the worse eye score. This is an experimental technical proxy, not gradability.

One shared **T=0.722602142** was fitted on 350 calibration patients in
331 containment components. Application is sigmoid(logit(clipped mean eye
probability)/T), never mean eye logits. The objective gives each component equal
mass and each of eight labels equal mass; mixed-label patients remain unchanged.
Calibration component-equal NLL **0.429812 → 0.419442** and patient Brier
**0.141013 → 0.139007** are **in-sample fit diagnostics**. Validation NLL/Brier
changes are conditional development evidence after model selection. No held-out
improvement or class-specific guarantee is claimed. All binary decisions remain
unchanged at p=0.5. H has only 10 positive calibration and 16 positive validation
patients/components; rare-class precision is limited.

| Fixed strategy | Raw Hamming AURC | Calibrated Hamming AURC |
|---|---:|---:|
| Confidence | 0.168652 | 0.168652 |
| + head uncertainty | 0.168612 | 0.168585 |
| + technical quality | 0.202406 | 0.200936 |

Lower is better. MC changes are tiny and paired descriptive intervals include
zero; adding quality worsened validation ranking. All three prespecified ablations
are retained. There is no post-hoc strategy/cutoff selection. Hamming error can
hide imbalance; any-label-error curves and per-class accepted support/FN rates
are also saved. [Results, all-eight support and plots](../results/reliability-v6/README.md).

Grad-CAM produced eight finite, nonzero maps for four seed-selected validation
patients, explaining the deterministic two-eye mean probability at the last
spatial stage. All positive labels are returned separately from the target.
Private overlays use canonical model-input geometry and are not lesion validation.

## Frozen candidate, limitations and recovery

[Candidate contract](../results/reliability-v6/candidate-contract.json) binds the
checkpoint, preprocessing, T, training-derived normalizer, predictors, source and
private immutable artifacts. Contract SHA256:
`a80dce8f27f12645539a0312a3ed96af0b5d10b5d7e1cd522af53b9347409c61`.
Private candidate artifact SHA256:
`4e1008bd0728e9ea21b21da8d5d45d5c0f96399b49c5f3b00d2d4efc707777a0`.

The research candidate is technically ready for a separately authorized held-out
evaluation with these fixed definitions. Final evaluation stayed sealed here.
Automatic ACCEPT is **not ready**: v4 has no supported independent policy fit
partition/risk contract, and rare-class evidence is inadequate for guarantees.
Deployment returns **REVIEW REQUIRED for every patient**. Calibration diagnostics
are in sample; validation was reused for selection. Quality harmed the development
ranking, and head-only variability omits backbone uncertainty. These limitations
cannot be resolved by choosing better-looking thresholds on these same labels.

All three private Kaggle stages completed; no monitor remains active. Sources,
nonces, report/artifact hashes, pass arrays, quality signals, parameter, plots and
Grad-CAM records were retrieved/verified. Large/sample artifacts remain outside Git.

| Stage | Pinned private job/version | Worker seconds |
|---|---|---:|
| analysis | `saisathwikregula/trustretina-phase6-analysis-ab40b8f2/1` | 14.628 |
| inference | `saisathwikregula/trustretina-phase6-inference-9a5132df/1` | 31.923 |
| inputs | `saisathwikregula/trustretina-phase6-inputs-0419699f/1` | 272.714 |

Observed GPU quota increase: **53.686 seconds**, paid scaling disabled.
Worker times exclude interpreter/provisioning overhead. [Compute receipt](../results/reliability-v6/compute.json).
Local checks: 128 tests passed with nine optional/environment skips; focused
actual-checkpoint Grad-CAM and exact Phase 5 reproduction checks also passed.
Phase 6 PR CI tracks the final source.

Fresh-task recovery uses the existing Kaggle SDK environment and account:

```sh
python scripts/phase6_experiment.py restore  # only if private registry is absent
python scripts/phase6_experiment.py retrieve
python scripts/publish_phase6_aggregates.py
python scripts/phase6_experiment.py export
```

Do not rerun init, completed inference, temperature fitting or training. The
[registry snapshot](../results/reliability-v6/run-registry.json) retains exact
IDs, versions and receipts. Private index: `artifacts/runner/phase6/PRIVATE_INDEX.md`.
Next: review the frozen research candidate, then explicitly authorize the bounded
held-out research evaluation. A clinical acceptance policy needs new independent
support and a prespecified risk/confidence contract; it remains out of scope.
