# Phase 11 — bounded nested loss study prepared; approval required

PR #20 is already merged at `75e9111`. V9/v10 reports were reviewed; no demo,
historical evaluation, calibration or exploratory-threshold deployment changed.
[Previous threshold evidence](STATUS-through-phase10-thresholds.md) is preserved.

The [prospective protocol](../NESTED_LOSS_V11.md), [configuration](../../configs/nested-loss-v11.toml)
and [concrete queue/support/cost report](../../results/nested-v11/README.md) define six fits:
three outer folds × ConvNeXt-Tiny cosine ratio-BCE reference versus square-root
ratio-BCE. Fresh ImageNet initialization in every fit; never reuse v9 weights.
Only the existing 2,275 training patients / 2,240 containment components are used.
Original v4 roles and both-eye/linked-patient grouping remain intact. No new cache,
duplicate investigation, GPU smoke, model inference or training was performed.

Each outer fold has a fixed 25% inner holdout for checkpoint selection; the
remaining 75% fits weights. Actual role counts are:

| Fold | Fit / inner / outer patients | H positives fit / inner / outer |
|---|---|---|
| 0 | 1137 / 379 / 759 | 33 / 11 / 23 |
| 1 | 1137 / 380 / 758 | 33 / 12 / 22 |
| 2 | 1137 / 380 / 758 | 34 / 11 / 22 |

All eight labels pass the prespecified positive/negative patient/component audit.
Every role has at least ten supplying components for each outcome. This is not
power or clinical adequacy evidence; H's inner support remains small. Source cache
metadata and hashes were reused; old validation/calibration/final payloads were
not opened. Worker code indexes only train tensors. Outer pixels are deferred
until complete training and strict best reload; pauses never assess outer data.

Checkpoint = maximum inner patient macro AUROC, earliest tie; six-epoch meaningful
patience, maximum 25. Arm comparison = mean paired outer macro AUROC; tie reference.
All use raw mean-eye sigmoid probabilities at .5; report macro AP/F1 and all-eight
precision/recall/support/TP/FP/FN, H detail and fixed paired component bootstrap.
No partial selection, new threshold fitting, calibration transfer or automatic refit.

**Estimated ceiling cost: 2.783 worker-hours; proposed strict free-quota cap: four
hours**, maximum six sessions including failures/resumes, each hard timeout 35
minutes plus five-minute guard. No seventh session reserve. Interruptions may
leave an incomplete matrix; preserve checkpoints and stop rather than extend the
budget. About 4.6623 free hours were observed, paid scaling disabled; recheck at
execution. All six runs remain queued, zero submission attempts, no approval file.

203 CPU tests: 194 passed / nine optional skips; compilation, source/manifest
integrity, real metadata support and diff checks pass. Mocked no-approval/no-API,
running-job deduplication and deferred-outer guards pass. Actual premature reporting
is refused. Automatic review rejected a live submission-guard test because a guard
failure could launch a job before approval; that live check was not executed.
GPU worker execution is untested and remains outside this turn's authorization.

Private authoritative state: `artifacts/runner/nested-v11/registry.json`,
`fold-manifest.json`, `PRIVATE_INDEX.md` and quota receipts. Public
[queue](../../results/nested-v11/queue.json), [preparation](../../results/nested-v11/preparation.json),
[support CSV](../../results/nested-v11/support.csv) and
[readiness](../../results/nested-v11/readiness.json) contain aggregates/hashes only.
Source-formatting-only plan rebinding is recorded with its unsubmitted predecessor;
no assignment/configuration changed. Current approval target is in the queue.

New fits prevent direct outer-patient weight/checkpoint/loss-weight leakage, but
prior adaptive development, same-source data and one seed still limit independence.
A fresh suitable external/temporal labeled cohort with adequate all-eight support,
particularly H, is needed to confirm improvement. Specify label adjudication,
representative sampling and clustered precision/power before collecting/evaluating;
no target accuracy is promised. Do not use the exposed historical final holdout.

**Only next authorization needed:** approve the exact six-fit plan and four-hour
free Kaggle cap. Then bind the actual user instruction to plan/source hashes and
six run IDs in private approval.json, inspect existing jobs/receipts/quota, and
advance sequentially under the same caps. No job runs or monitoring continues now.
Original demo and REVIEW REQUIRED remain unchanged.
