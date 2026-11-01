# Phase 10 — CPU-only exploratory threshold analysis complete

PR #19 is already merged at `f5f50fb`; its CI `41291834977` passed. Review confirmed the full four-run report preserves negative findings and correctly limits the conclusions to reused development data. No human PR review submissions were recorded. [Phase 9 status/history](history/STATUS-through-phase9-results.md) preserves the training results and recovery instructions.

The [v10 protocol](THRESHOLD_V10.md) and [configuration](../configs/threshold-v10.toml) were committed at `1d1516f` before any threshold computation; implementation/tests were committed at `14f6382` before execution. The user authorized existing validation predictions for this separate exploratory analysis; the original training-OOF-only helper and frozen v9 protocol remain unchanged.

## Actual development findings

Selected ConvNeXt cosine epoch 13, raw mean-eye probabilities, same 525 patients / 518 components. Four fixed label-stratified group folds (seed 2026), assessment sizes 132/131/131/131; every patient assessed once and no containment-component crossing. Thresholds maximize fitting-fold per-class F1 on .05:.05:.95; exact ties nearest .5, then higher. Fit support below 10 supplying groups per outcome retains .5; no class required a fallback.

| Procedure | Macro F1 | Micro F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| Fixed .5 | 0.5926 | 0.5714 | 346 | 243 | 276 |
| Held-out threshold CV | 0.6055 | 0.5880 | 436 | 425 | 186 |

CV reduced missed positives by 90 but added 182 false positives. Conversely, .5 avoids those 182 false alarms at the cost of 90 more missed positives. N/D/G/M/O F1 rose, C F1 fell, A/H were unchanged. Exact match fell .4152 → .2952 and labelwise accuracy fell .8764 → .8545; neither is a clinical diagnostic-accuracy claim.

H threshold was .50 in every fit, **not a fallback**: each fit had 12 positive-supplying groups and each assessment fold had four positive patients. Pooled H TP/FP/FN remain 3/1/13, precision .75, recall .1875, F1 .30. Agreement on this coarse grid with sparse overlapping fitting samples is not robust clinical threshold evidence. Full all-eight precision/recall/F1/counts, fold support and threshold quartiles/ranges are in the [generated report](../results/threshold-v10/RESULTS.md), [aggregate JSON](../results/threshold-v10/aggregate.json) and [CSV](../results/threshold-v10/per-class.csv).

## Preservation, checks and recovery

No model training/inference, new GPU job, calibration fit, acceptance policy, new data setup or paid resources. Historical final and calibration datasets/manifests were not opened. No old temperature was applied. Five frozen demo/contract artifacts remain byte-identical; original demo and REVIEW REQUIRED behavior are unchanged. No all-development threshold vector was refitted or exported for deployment.

Pinned inputs: `saisathwikregula/trustretina-round2-v9-convnext-cosine-ede050b3/1`; checkpoint SHA256 `af33eb06fb2de21929e4dd3ec3959c0fbc58890d126a50c2e0d2b34bafaa0d49`; patient prediction SHA256 `e3f03f4fda4439aee575b12fd1cebd47056d97143f8a7bef3decc0b5ae51f44a`. Source/nonce/report/all-output integrity and saved mean-eye aggregation were verified. The private fold assignments, OOF decisions, fitting curves, provenance and completion hashes live in `artifacts/analysis/threshold-v10/attempt-1/`; original private Kaggle artifacts remain recoverable.

CPU analysis before report rendering: 1.305 wall seconds. Scientific suite: 194 tests, 185 passed / 9 optional skips. Six new focused tests cover exact ties, sparse-group fallback, deterministic mixed-label grouping, unchanged own-fold thresholds after held-out labels/scores change, cross-fold leakage rejection and analytical 2-D metrics. Actual saved decisions were separately checked against scikit-learn precision/recall/F1/confusion counts within 1e-12. Compilation and diff checks pass.

Re-run `python scripts/analyze_thresholds_v10.py` in the existing CPU/report environment to verify/reuse a completed analysis; it does not fit again or overwrite mismatched artifacts. Required original selected inputs can be restored read-only using [Phase 9 recovery](ROUND2_RECOVERY.md); reuse the pinned selected run, never re-submit it. Public provenance records input/output hashes; sample-level outputs stay private.

## Concrete next step

Do not install the exploratory thresholds or replace the demo. Before any new threshold/deployment claim, prospectively define error costs/objective and obtain development predictions independent of checkpoint selection, e.g. training-only containment-aware out-of-fold model predictions in a separately authorized study. Freeze the complete new pipeline and assess it on a fresh suitable independent cohort with adequate H support. Calibration for the new checkpoint remains a separate prospective fit; never transfer the old temperature. No additional run or fit is authorized or queued here.

Each threshold-assessment fold was untouched by its threshold fit, but all validation labels already influenced model/epoch selection. This fixed-model CV cannot remove that adaptive selection bias and does not provide independent confirmation. No clinical recall target or automatic acceptance policy is inferred.
