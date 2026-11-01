> Historical pre-approval preparation record, preserved below. The six-fit study is now complete. See [actual results](RESULTS.md), [execution audit](EXECUTION.md), [provenance](execution-provenance.json) and [current status](../../docs/STATUS.md). Statements about pending approval/no execution below describe preparation only.

# Nested loss study v11 — prepared, not executed

[Prospective protocol](../../docs/NESTED_LOSS_V11.md) · [fixed configuration](../../configs/nested-loss-v11.toml) · [readiness evidence](readiness.json) · [all-eight support CSV](support.csv)

No GPU job or model inference was launched. This is metadata feasibility and a concrete queue, not new predictive-performance evidence. Original demo, v10 thresholds, calibration and historical final evaluation remain unchanged.

## Exact matrix

| Run ID | Fold | Loss | State |
|---|---:|---|---|
| fold0_ratio_bce | 0 | ratio_bce | queued; awaiting approval |
| fold0_sqrt_ratio_bce | 0 | sqrt_ratio_bce | queued; awaiting approval |
| fold1_ratio_bce | 1 | ratio_bce | queued; awaiting approval |
| fold1_sqrt_ratio_bce | 1 | sqrt_ratio_bce | queued; awaiting approval |
| fold2_ratio_bce | 2 | ratio_bce | queued; awaiting approval |
| fold2_sqrt_ratio_bce | 2 | sqrt_ratio_bce | queued; awaiting approval |

Reference positive weight = fitting-patient negative/positive ratio; alternative = its square root. Only this loss factor changes within each pair. All start fresh from the same official ImageNet ConvNeXt-Tiny weights, not the v9 checkpoint that already saw all training patients.

## Actual training-only support

| Fold | Role | Patients | Components | H positive / negative patients | H positive / negative components |
|---|---|---:|---:|---:|---:|
| 0 | fit | 1137 | 1122 | 33 / 1104 | 32 / 1092 |
| 0 | inner | 379 | 373 | 11 / 368 | 11 / 363 |
| 0 | outer | 759 | 745 | 23 / 736 | 23 / 722 |
| 1 | fit | 1137 | 1117 | 33 / 1104 | 33 / 1084 |
| 1 | inner | 380 | 377 | 12 / 368 | 12 / 366 |
| 1 | outer | 758 | 746 | 22 / 736 | 21 / 727 |
| 2 | fit | 1137 | 1123 | 34 / 1103 | 34 / 1089 |
| 2 | inner | 380 | 368 | 11 / 369 | 10 / 360 |
| 2 | outer | 758 | 749 | 22 / 736 | 22 / 728 |

All eight labels have at least ten positive- and ten negative-supplying components in every role. That feasibility floor does not establish adequate statistical power. Both eyes and containment components stay together; every training patient has exactly one outer assessment. H remains sparse and inner checkpoint selection may be noisy. No original v4 partition was changed.

## Selection and reporting

Maximize all-eight inner patient macro AUROC for checkpoint selection, earliest exact tie; patience six meaningful improvements, min delta 1e-4, max 25 epochs. Common cosine 1e-4→1e-6, AdamW/FP32, batch 16, weight decay .01, seed 2026, 224-pixel original preprocessing, no augmentation. Mean-eye sigmoid probabilities; fixed p≥.5 only. Outer inference occurs only after training completes and best strict reload; paused runs do not touch outer images.

Primary contrast is mean paired outer macro AUROC difference; exact tie retains reference. Report macro AP, macro/micro F1, all eight precision/recall/F1/TP/FP/FN/support, per-fold H detail, and paired component bootstrap intervals. No partial winner; no threshold/calibration fitting or automatic full-data refit. Predictions and large artifacts stay private. Earlier adaptive choices, same-source data, reduced fitting size and one seed still limit independence.

## Compute and execution boundary

Measured reference whole-worker time: 2026.674 s / 19 epochs. The estimate scales fitting and selection computation separately and retains 9.762 s fixed overhead per epoch, plus a 180-second per-fit allowance and one outer pass.

Estimated worker cost at all six 25-epoch ceilings: **2.783 hours**, about 27.8 minutes per fit. This approximation is not a billing or completion guarantee. Early stopping can reduce time; slower hardware/IO can cause a pause.

Proposed strict cap: **four free Kaggle quota hours**, at most six submitted sessions total including failures/resumes, each ≤35 minutes plus a five-minute charged overhead guard (six ×40 minutes). There is no seventh session reserve. A paused/failed fit can make the matrix incomplete; preserve progress and stop at the cap rather than silently extending it. No paid scaling or fallback.

Observed free quota: 4.662295 h at 2026-11-01T10:08:20.149397+00:00; paid scaling disabled, zero new jobs. Recheck before every future submission.

Frozen fold manifest SHA256: `e8f3e7a6e44880c2a5d09dac08619df029d650658d720b26b15352721ca844a5`.
Plan SHA256: `55720cb1ba2da127af30dec04ebc10b35d7b579e6e99435985a9059e61ccba34`.
Worker/source SHA256: `1512109180685148b7aca4b9d1fd4160eef434f4c30d65721a3aa9d6f4790eab`.
Recorded implementation revision: `93300d694dea55d4ccba83027a7b8aff7a43ee43`.

The unsubmitted precursor plan is preserved privately; a source-formatting correction changed the plan/source digest without changing CV assignments or settings. The current hashes above are the only approval target.

## Verification and recovery

203 CPU tests: 194 passed, nine optional skips. Checks cover linked-patient separation, outer coverage, K=4 equivalence to the frozen allocator, loss weights, exact approval/budget guards, no duplicate running-job submission, deferred/one-time outer access, checkpoint provenance and paired reporting/bootstrap. The GPU worker itself is unexecuted. A live submission-guard check was rejected by automatic review before execution because guard failure could consume quota; mocked no-API guard tests passed. Premature report generation was actually rejected for the incomplete matrix.

Private state: `artifacts/runner/nested-v11/registry.json`, `fold-manifest.json`, quota receipts and `PRIVATE_INDEX.md`. No approval file exists. `status` is local; `check-quota` is read-only. `restore` recreates only identical hash-matching CV metadata from the saved train rows, never the dataset/cache; `retrieve` downloads pinned terminal attempts without submission. Future `advance --submit` requires an actual new user approval bound to this exact plan/source/six IDs/14,400-second cap.

## Confirmation data

A new suitable independent labeled cohort must exclude overlapping patients/copies across all prior cohorts, include paired eyes and authoritative all-eight patient labels, and follow a prospectively frozen labeling/assessment protocol. Prefer representative temporal/external-site/device sampling; do not select cases for favorable errors. Record co-occurrences and adequate positive/negative support, particularly H. Rare-class enrichment needs sampling weights and does not estimate deployment prevalence/PPV/calibration unweighted.

Choose size with a clustering-aware precision or paired-power calculation. Roughly 97 independent positive patients per label is only an illustrative worst-case binomial requirement for ±.10 recall precision at nominal 95%, before clustering/multiplicity/design inflation—not a clinical recall target or accuracy promise. Final required total depends on prevalence and design. Freeze the complete new pipeline before accessing that cohort; never reuse the old exposed final holdout to confirm a new choice.
