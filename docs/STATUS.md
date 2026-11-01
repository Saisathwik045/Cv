# V11 approved execution in progress

PR #21 was already merged at `3b9856b`; CI `41296752336` passed. The user explicitly
approved plan `55720cb1ba2da127af30dec04ebc10b35d7b579e6e99435985a9059e61ccba34`
and worker/source `1512109180685148b7aca4b9d1fd4160eef434f4c30d65721a3aa9d6f4790eab`.
Both were verified from latest merged main before submission. All six settings,
private training-only fold assignments and original model/data behavior are fixed.

The first job is `saisathwikregula/trustretina-nested-v11-fold0-ratio-bce-2ab221ef/1`.
The first fit completed at 12 epochs. The second is running as
`saisathwikregula/trustretina-nested-v11-f0-sqrtbce-644b376e/1`; four remain queued.
A pre-session SaveKernel400 on its original 52-character provider name was
recovered after inventory/exact-version/quota checks. Only the provider name
changed; source/configuration and the existing reservation were retained.
Rejected request/metadata and recovery evidence are preserved, with no version
or GPU charge for that rejected request. Do not infer completion from this static paragraph:
[exported queue](../results/nested-v11/queue.json) and private
`artifacts/runner/nested-v11/registry.json` are authoritative. The in-task bounded
driver is active (exec session 62473). Do not start a second driver. Inspect
`driver-latest.json`, saved receipts and actual remote versions before recovery.

Execution: sequential; at least 180 seconds between reconciliations; maximum six
submitted sessions including failures/resumes, hard timeout 2100 seconds each,
300-second charged overhead reserve, strict 14400 free Kaggle quota seconds total.
No paid scaling or seventh session. Fresh API used/reserved/available GPU quota
is captured before each push; actual counter deltas drive accounting, not worker
time. An added conservative rate guard can stop later submissions if accelerator
billing differs from the prepared reservation. It cannot enlarge the budget or
change approved code/configuration. First pre-submit quota was 16784.262 seconds
free, paid scaling false; original baseline used was 4815.738 seconds.

The worker/source, controller and prespecified reporter are unchanged. New
orchestration/accounting and partial-audit helpers live under scripts, outside
the approved worker hash. Best/last checkpoints, RNG/optimizer/scheduler, histories,
logs, predictions, source/nonce receipts and checksums stay private in exact
Kaggle versions and ignored artifacts. Outer evaluation is deferred until complete
training and strict best reload; paused sessions never assess outer images.

After all six complete, run `python scripts/report_nested_results.py
--bootstrap-batch 100` to checkpoint/resume the fixed 2000-replicate paired CPU
analysis. `python scripts/summarize_nested_execution.py` verifies available
artifacts and can report an incomplete matrix without selecting a winner.
Do not alter thresholds, calibration, demo or historical final evaluation; do not
refit on all data. No old validation/calibration/final payload is needed.

Private approval is `artifacts/runner/nested-v11/approval.json`; do not recreate
or edit it. `TASK_RESUME.json`, `PRIVATE_INDEX.md`, accounting snapshots and
per-attempt quota/source/submission/verification receipts preserve exact state.
When driver returns/stops, no continued monitoring is promised. Recover existing
jobs before any submission; if caps or accounting prevent continuation, publish
completed evidence and declare the matrix incomplete.

[Original prospective protocol](NESTED_LOSS_V11.md),
[prepared support/budget report](../results/nested-v11/README.md) and
[pre-approval status](history/STATUS-through-v11-plan.md) preserve the design.
Previous adaptive choices, sparse H and one seed limit independence; new suitable
independent labeled evaluation is still required. REVIEW REQUIRED remains default.
