# V13 matched resolution study prepared — training not authorized

PR #23 is already merged at `d94fbda`; CI41311898604 passed all seven jobs.
[Preparation/approval target](../results/resolution-v13/README.md),
[frozen protocol](RESOLUTION_V13.md), [configuration](../configs/resolution-v13.toml),
[verified native/parity evidence](../results/resolution-v13/prerequisites.json).

One GPU-disabled private CPU job, `saisathwikregula/trustretina-res-v13-inputs-767673ae/1`,
completed and was retrieved/source/nonce/checksum verified. All4550 original TRAIN
images matched frozen raw hashes; local independent archive rehash also passed.
All4550 rebuilt224 inputs matched the saved cache exactly (zero differing uint8
channel values). The320 tensor was built directly from originals; original source,
rows and old cache remain immutable. Native tar, both arrays, receipts and ImageNet
weights are durable in that pinned private output; patient data stays outside Git.
No new raw source upload or GPU smoke. CPU worker763.25s; GPU-quota charge0s.

Authoritative ODIR material confirms patient-level H=hypertension and O=other
abnormalities/diseases, using both-eye information. Precise H criteria, O taxonomy
and eye-localized truth remain unresolved; benchmark-target ablation is not a
clinical claim. No relabeling. See the source-access limitations in readiness.

Six queued runs: fold0/1/2 × res224/res320, fresh ImageNet ConvNeXt-Tiny, ratio BCE,
seed2026, same fixed training-only inner/outer containment folds. Common8-image
microbatches with2-step accumulation, effective16, FP32 AdamW/cosine, max25 epochs,
AUROC inner checkpoint selection and patience6/min delta1e-4. V11 reference runs
cannot substitute because their16-image/no-accumulation execution differs. All
five factories and both historical losses are preserved. Outer assessment occurs
only after complete fit and strict best reload; fixed mean-eye sigmoid p≥.5.

Plan: `0fa597609d0390e7414865ed9507cd97e214c99945abd4c1cc095a9924b1cf2a`.
Worker/source: `e8a8df1bbca95b14109f4e16ca31080e9dd9afe4746d42af5e228b96c0a5e80c`.
Configuration: `fcc8c73b752551a5d5b860b3d134e2828b78ea7a4a5aa41c48549b3a717ae0b4`.
Exact matrix, all-eight support and remaining identities are in preparation README.

Proposed strict cap21600 free GPU-quota seconds / max6 sessions, including every
failure/resume/overhead; reservations2700s for224 and4500s for320. Estimate4.23
worker-hours,5.50 with allowance, not a new measured throughput/memory guarantee.
Observed free quota12670.775s (3.5197h), GPU reserved0, paid scaling disabled.
**First-job gate is blocked** until the full six-hour reservation is actually free.
Provider reports refresh7Nov2026 at05:30IST; recheck, do not assume future quota.
SDK does not expose guaranteed per-shape billing. Counter-based/rate guards and
bounded sequential reconciliation stop rather than use paid resources or exceed
caps. Zero training submissions; no approval record or background monitor.

225 CPU tests:216 passed, nine optional skips. Accumulation/remainder, stochastic
checkpoint resume, source/archive integrity, approval/budget/recovery guards and
both resolution tensors are covered. Synthetic official-ImageNet CPU forwards
were finite at224/320; no new patient inference or GPU memory experiment. Compiler
and diff checks pass. New PR CI is reported separately, never inferred from local
checks or earlier commits.

**Next action:** review preparation PR and approve the exact six-fit plan/source,
run IDs and21600-second/six-session cap if desired; wait for verified full free
quota before any training. Do not refit all data, deploy thresholds, transfer the
old temperature, change demo or access historical final evaluation/calibration.
The original demo/temperature/normalizer/contract hashes and old worker source
match prior records; REVIEW REQUIRED remains default.

Private state: `artifacts/resolution-v13/TASK_RESUME.json`, `PRIVATE_INDEX.md`,
preparation receipts/artifacts and experiment registry. Fresh-task recovery uses
`python scripts/restore_resolution_state.py` with existing secure Kaggle access;
it retrieves pinned outputs and original fold metadata, creates no approval and
submits no jobs. Conflicting/newer state is never overwritten. No new dataset,
partition search, duplicate review or historical evaluation was performed.
[Prior audit](history/STATUS-through-v12-audit.md) is preserved.
