# V11 six-fit study complete

PR #21 was already merged at `3b9856b`; all seven CI jobs passed. The exact
user-approved plan/source hashes were verified before execution. All six fresh
ImageNet ConvNeXt-Tiny fits completed, with no checkpoint resumes or seventh GPU
session. [Results and paired intervals](../results/nested-v11/RESULTS.md),
[execution audit](../results/nested-v11/EXECUTION.md) and
[accounting/artifact provenance](../results/nested-v11/execution-provenance.json)
record actual evidence. The original architecture and all five factories remain.

| Fold | Ratio BCE epochs / best | Sqrt-ratio BCE epochs / best |
|---|---:|---:|
| 0 | 12 / 6 | 9 / 3 |
| 1 | 12 / 6 | 8 / 2 |
| 2 | 15 / 9 | 10 / 4 |

Actual settled free Kaggle GPU-quota charge: **4113.487 seconds (1.142635 hours)**,
below the 14400-second cap. Six versioned GPU sessions; paid scaling disabled,
zero GPU time reserved at completion. Separate summed worker time was
3953.290 seconds; worker duration is not used as charged quota. Before every
submission, actual used/reserved/free quota and accelerator accounting were
checked against the existing reservation. Final owned-job inventory matches
exactly the six recorded versions. No GPU job or background monitor is active.

One additional SaveKernel request returned HTTP400 before a session/version was
created. Inventory, exact-version and quota recovery checks found no job or
charge. Only the provider name was shortened; approved scientific source,
configuration, nonce and reservation were unchanged. Identifier validation is a
possible cause, not proven because the initial response body was unavailable.
The failed request and recovery evidence are retained privately; this was not a
seventh GPU session or a repeated completed fit.

All six best/last checkpoints, histories, aligned image/patient probabilities,
labels/groups, source/nonce receipts, dependency records and output hashes were
retrieved and verified. The executable embedded worker was independently decoded
and matched the approved SHA256. Inner-only best/stop selection was replayed;
outer rows/groups match frozen assignments and mean-eye aggregation was
recomputed. Original demo checkpoint/temperature/normalizer/contract hashes match;
historical final and calibration payloads were not opened. No threshold fitting,
calibration transfer, full-data refit or demo promotion occurred. REVIEW REQUIRED
remains unchanged.

Square-root weighting is the prespecified future-study candidate: mean outer AUROC
0.83549 versus 0.83451 (difference +0.00098; conditional 95% interval −0.00930
to +0.01058). Mean macro F1 falls from 0.53901 to 0.51359. Both arms miss 58/67
H positives. This does not establish improvement. See the [interpretation and
all-eight pooled counts](../results/nested-v11/DISCUSSION.md).

## Recovery and next action

Private state is `artifacts/runner/nested-v11/registry.json`, `PRIVATE_INDEX.md`,
`TASK_RESUME.json`, exact per-attempt receipts and pinned private Kaggle versions.
The public [queue](../results/nested-v11/queue.json) exposes run/version state
without patient records. A fresh-download proof verifies remote artifact recovery.
Do not start the submission driver: the approved matrix is complete.

`python scripts/nested_experiment.py retrieve` retrieves existing terminal
versions without submitting jobs. `python scripts/report_nested_results.py
--bootstrap-batch 2000` reproduces/resumes the fixed 2000-replicate CPU analysis;
`python scripts/plot_nested_comparison.py` renders aggregate figures. Use the
recorded scientific dependencies and verified private metadata. Missing registry
recovery uses the documented hash-exact metadata restore, never new splits/cache.
No credentials, predictions, patient manifests or checkpoints belong in Git.

208 CPU tests ran: 199 passed and nine optional skips. No new GPU smoke, duplicate
review, split generation or historical evaluation was performed. Earlier adaptive
choices, same-source data, one seed, only three outer folds and sparse H limit
inference. Conditional bootstrap intervals omit training/selection variability.
A candidate loss for further study is not independently confirmed improvement.
The next research step requires a separately approved plan and a fresh suitable
independent labeled cohort; there is no automatic refit or deployment step.

[Prospective frozen protocol](NESTED_LOSS_V11.md),
[historical preparation/support report](../results/nested-v11/README.md) and
[pre-approval status](history/STATUS-through-v11-plan.md) preserve the design.
