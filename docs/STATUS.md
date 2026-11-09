# V13 approved execution in progress

The user explicitly authorized the frozen six-fit matched224/320 experiment on
9November2026, within21600 free Kaggle GPU-quota seconds and max6 sessions,
including failures/resumes/overhead. No paid resources or automatic all-data refit.
[Queue](../results/resolution-v13/queue.json) and private
`artifacts/resolution-v13/experiment/registry.json` are authoritative.

Exact plan `0fa597609d0390e7414865ed9507cd97e214c99945abd4c1cc095a9924b1cf2a`,
worker/source `e8a8df1bbca95b14109f4e16ca31080e9dd9afe4746d42af5e228b96c0a5e80c`
and configuration `fcc8c73b752551a5d5b860b3d134e2828b78ea7a4a5aa41c48549b3a717ae0b4`
were verified unchanged. Current private authorization is `experiment/approval.json`;
never replace it or invent another approval.

Fresh-task recovery downloaded the actual pinned preparation output, source and
nonce receipts; all artifact hashes and original frozen fold spec verified. No
images,320 cache, split assignments or duplicate investigation were rebuilt.
Remote owned-job inventory had54 entries and no v13 training job. Observed initial
GPU quota was21600s free, zero used/reserved, paid scaling disabled.

Fold0 completed at both resolutions:224 ran13 epochs (best7),320 ran11
epochs (best5); both strict reloads and artifact checksums passed. Settled quota
used before fold1_res224 was2137.812s; paid disabled, reserved0. Fold1_res224
started as `saisathwikregula/trustretina-res-v13-f1-224-19e7d4a3/1`.
The first version is `saisathwikregula/trustretina-res-v13-f0-224-fb0c42fb/1`.
Do not infer later completion from this paragraph. `driver-latest.json` and the
registry list exact versions/status; first fit was submitted, five initially queued.
One driver (exec session63834) is active. Do not launch duplicate drivers or
submission commands. The orchestration helper calls the unchanged frozen
controller, observes quota immediately before each push and waits at least180s
between reconciliations. Every terminal version is retrieved/source/nonce/output-
hash verified. Reservations2700s for224 and4500s for320 enforce21600 total/max6;
actual provider counter intervals are separate from worker timing. Stop on
accounting, free-quota, paid-scaling, session-cap or ambiguous-state blockers.

Shared microbatch8/accumulation2, fresh ImageNet ConvNeXt, ratio BCE, grouped
training-only fit/inner/outer data, fixed mean-eye predictor and p≥.5 are unchanged.
No source/config/protocol/reporter identity changes. The old demo, historical
final evaluation, calibration, thresholds, five factories and both losses remain
unchanged. REVIEW REQUIRED remains default.

After all six verified complete, the frozen CPU reporter
`python scripts/report_resolution_results.py --bootstrap-batch 2000` runs the prespecified2000 paired component
bootstrap replicates, resumably. An incomplete matrix must not select a winner.
`scripts/audit_resolution_execution.py` statically checks the actual embedded
worker and replays inner-only selection, fit-only weights and outer alignment.
Its partial audit never selects a candidate. Patient-level artifacts stay private.

Private state/receipts: `artifacts/resolution-v13/TASK_RESUME.json`, `RESUME.md`,
`experiment/registry.json`, per-attempt source/submission/verification and driver
snapshots. If interrupted, inspect existing driver/remote versions first; never
repeat CPU preparation or blindly resubmit an ambiguous job. No monitoring is
promised after a task ends; exact current lifecycle is saved privately.
[Historical preparation/evidence](../results/resolution-v13/README.md) and
[prior status](history/STATUS-through-v13-preparation.md) remain intact.
