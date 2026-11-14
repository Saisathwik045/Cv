# V15 calibration/reliability software — synthetic verification only

V15 source development starts from remote main
`badbf8af8ef3c0ac57a942f70e594e9b8634a419`, on
`phase/15-calibration-reliability`. [Prospective protocol](CALIBRATION_RELIABILITY_V15.md)
and [execution readiness](V15_EXECUTION_READINESS.md) define the separate real-data
permission boundary. No real V15 scientific result is reported.

Implemented CPU analysis for both frozen V14 candidates: trusted artifact/checkpoint
SHA256 checks, original role-manifest and patient/component/eye/class alignment,
separately gated shared-temperature fitting, NLL/Brier/10-bin ECE, per-class
reliability diagrams/counts/support, saved 20-pass head-only MC and TRAIN-only
resolution-specific technical quality, paired FP/FN/precision/recall/F1 differences
and descriptive tied-score risk-coverage/component intervals. The unchanged shared
algorithm is reused; no threshold or acceptance policy is fitted. All predictions
remain REVIEW REQUIRED. No historical final access, demo change, private dataset
or checkpoint access, model inference, Kaggle submission or real-data fitting.

Validation: all 28 V15 synthetic tests passed. The complete CPU suite ran 277
tests: 268 passed, nine optional photographic tests skipped, no failures/errors.
This includes existing toy optimizer/resume fixtures, with no real-model training.
Tests ran from a temporary source snapshot with CUDA hidden and test-process
network transport disabled. Readiness/synthetic CLI, phase0/phase1 config,
declarative plan, v3/v4 development checks, environment report, compilation,
`pip check`, and `git diff --check` passed. Original V14 controller freeze
verification passes byte-for-byte; `src/`, old scripts, configs, requirements,
results and demo remain unchanged. PR CI is pending publication; see the PR for
observed status. Synthetic fixture values are not research results.
Real-data execution remains blocked on separate explicit authorization, a reviewed
hash-bound manifest and verified permitted private artifacts/role records. BRSET
access and independent all-eight compatibility blockers persist. Existing optional
photographic dependencies and Kaggle recovery tooling remain absent locally.

---

# V14 two frozen fits complete — reused-validation development only

Merged PR27/main3b07a7634f3e6530d66b2ec41c70d837893992c6; approved plan/source/config/
protocol/dependency/input identities preserved. No preparation, duplicate review
or v13 training was repeated. Exact two fits completed sequentially with one
session each, zero failures/resumes, and no paid scaling.

| Fit | Epochs / best | Macro AUROC | Macro AP | Macro F1 | Micro F1 |
|---|---:|---:|---:|---:|---:|
| ConvNeXt320 | 11 / 5 | 0.8596 | 0.5993 | 0.5371 | 0.5532 |
| ConvNeXt224 | 11 / 5 | 0.8480 | 0.5812 | 0.5332 | 0.5290 |

[Complete all-eight precision/recall/F1 and TP/FP/FN](../results/candidate-v14/RESULTS.md)
and [CSV](../results/candidate-v14/validation-per-class.csv).320 leads aggregate
validation metrics but is not uniformly better: H FP6 vs21 but FN12 vs11; O FP119
vs118, FN53 vs61; D/A/M misses107/14/4 vs130/7/5. C FP99 vs30 and F1 0.3827 vs 0.6222;
A recall0.44 vs0.72; G recall0.50 vs0.5625. Exact match0.2990 vs0.3143. No independent
confirmation or clinical superiority; no automatic promotion/refit/demo change.

Private pinned jobs:
- saisathwikregula/trustretina-v14-fit-320-7bbe5ade/1
- saisathwikregula/trustretina-v14-fit-224-779a38cd/1

All 34 output-file hashes passed, including best/last checkpoints, aligned raw eye/
patient predictions, histories, dependency records,20-pass seeds and quality arrays.
Both strict best reload/state-invariance/post-freeze inference checks passed.
Small artifact/prediction recovery was independently verified from both versions;
large checkpoints and patient records remain private outside Git.
[Artifact hashes](../results/candidate-v14/verified-artifact-registry.json),
[fresh retrieval](../results/candidate-v14/fresh-retrieval-proof.json),
[run registry/receipts](../results/candidate-v14/frozen-plan.json).

[Actual quota accounting](../results/candidate-v14/gpu-time.json):3826.407s(1.062891h)
observed account delta vs14400s cap; two sessions vs maximum3, no resume. Baseline
timeUsed6926.895s and refresh2026-11-14T00:00Z retained throughout. Final used10753.302s,
free10846.698s, reserved0, paid disabled. Worker sum3767.781s is distinct from billed
quota. Internal conservative reservations12000s. Foreground queue driver exited;
no background monitoring or further GPU session is active.

No historical final access, external evaluation, temperature/threshold fitting or
working-demo modification. Calibration output metadata has no labels; inference
ran only after checkpoint freeze/strict reload. REVIEW REQUIRED stays unchanged.

Next executable research step requires a separately authorized, prospectively
recorded calibration/reliability analysis for these new checkpoints, using saved
post-freeze inputs/predictions. Do not reuse the original demo's temperature or
threshold experiments. BRSET remains a separate credentialing/CITI/personal-DUA/
label-compatibility blocker; no agreement/form submitted. Independent all-eight
confirmation remains unavailable. GPU training queue is complete; do not resubmit.
