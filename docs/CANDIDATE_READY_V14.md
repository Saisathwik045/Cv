# V14 executable worker/input contract — GPU execution not authorized

This implementation completes the CPU-readiness stage authorized after merged
PR26/main `f722cb51167b1694802ffdbbe297866b7e8db2a9`. It preserves the
[CANDIDATE_FREEZE_V14.md](CANDIDATE_FREEZE_V14.md) two-fit scientific design and
its four-hour/three-session budget. The original v13 study, source/protocol,
224 reference, original demo, historical final, fitted temperature/normalizer and
thresholds remain unchanged. REVIEW REQUIRED remains default.

## Completed CPU scope and storage

Only one private GPU/TPU-disabled CPU preparation session is allowed, provider
timeout3,600s. It reuses the frozen224 validation/calibration tensors exactly and
constructs only native320 validation/calibration tensors after frozen raw-byte
SHA256 checks. No TRAIN image preparation, split generation, duplicate review,
training, temperature/threshold fit or external inference is part of readiness.
All selected source paths belong to the existing v4 validation/calibration
allowlists. Historical final records/images are never selected or opened.

Existing verified artifacts stay in place. Free `/tmp/trustretina-v14-private`
storage is private(mode0700), separately capacity/write-tested. It is ephemeral;
its durable authority is the exact pinned private Kaggle output versions and
receipt/source/report/output hashes. No deletion or paid storage is allowed.
Actual TRAIN224/TRAIN320 remain in their verified original cache, rather than
being recopied into scratch. Fresh-task recovery selects existing actual inputs,
not metadata substitutes, and never submits preparation or training.

## Data-role and optimization contract

Two runs only, in this order:

1. `v14_candidate_res320_seed2026`: worker timeout7,500s, reservation7,800s.
2. `v14_reference_res224_seed2026`: worker timeout3,900s, reservation4,200s.

Both use frozen v4 TRAIN2,275 patients for optimization and validation525 for
checkpoint selection; calibration350 contributes images only after a selected
checkpoint has been persisted and strictly reloaded. Exact release/role hashes,
actual tensor/index/row hashes, ImageNet initialization, dependency/Docker revision
and source digests are bound by the frozen plan. Every patient retains both eyes;
containment components must not cross roles. Counts and all-eight positive/
negative support must match the frozen aggregate records. Original labels remain
patient-derived targets, without relabeling or clinical adjudication.

Only TRAIN patient negative/positive counts determine ratio BCE weights. The
existing five factories and both loss implementations are retained. Both fits
start fresh verified official ImageNet V1, with the original eight independent
logits and trained p0.2 dropout head. Microbatch8/accumulation2, effective16 with
exact partial-window sample weighting; FP32 AdamW1e-4, weight decay.01, cosine
minimum1e-6 over25 epochs, seed2026, no augmentation. Both backbone and head are
fine-tuned as in v13. Validation patient macro AUROC selects checkpoints, earliest
exact tie; meaningful improvement1e-4/patience6 controls early stopping. Maximum25
epochs is per fit across all sessions, not per resume. No refit on validation.

The worker mounts only existing TRAIN-only preparation output and the new v14
CPU output. It never mounts original raw datasets or the old Phase6 inputs.
Calibration rows in the new bundle have labels removed; the original labeled
metadata is retained separately in its pinned old private source for a future
separately authorized calibration analysis. CPU readiness may audit that metadata
for lineage/support, but it cannot influence optimizer weights/checkpoint choice.
Training/validation rows keep the original full targets. No zero/fake calibration
labels are exported or interpreted as negatives.

Before post-fit calibration pixels are opened, strict best reload succeeds and
`selected-checkpoint.json` records the best SHA256, epoch, data/source/configuration
provenance, fixed patient predictor and threshold. Only then are raw logits,
mean-eye probabilities and20 trained-head MC samples produced for calibration.
Validation outputs include original labels/IDs; calibration outputs have no
labels or metrics. Patient probabilities use float64 means of the two eye sigmoid
outputs, never mean logits. Validation/calibration eye order is sorted patient ID
then eye; deterministic heads run in the declared microbatch while MC uses the
whole ordered cached-feature matrix per pass. Seeds62026..62045(validation) and
72026..72045(calibration); normalization/stochastic depth remain eval. No added
untrained dropout. Model state is compared before/after inference. Quality outputs
are TRAIN/validation/calibration technical features only; no temperature, normalizer,
threshold or accept/review policy is fitted by this worker.

Best/last checkpoints are saved atomically every completed epoch with optimizer,
scheduler and all RNG/loader state. Paused sessions have no calibration inference.
Resume requires identical provenance/dependencies/input hashes and verified
progress/best/last/history bytes. Strict tensor-only reload restores state; epochs
and scheduler continue toward the original25 horizon. An interrupted post-freeze
stage is not treated as a completed fit unless strict reload, invariance and
post-freeze calibration evidence are all present. It may resume from its valid
last epoch within the same global budget, without another optimization epoch if
training already finished.

## Quota, sessions and recovery

No GPU job is authorized by CPU readiness. The future approval must bind the exact
plan, worker, configuration and input hashes, both run IDs,14,400 free GPU-quota
seconds, maximum3 sessions and acknowledge that external confirmation is not ready.
The full four-hour free quota must be observed immediately before the first
submission. Before every push, check fresh(≤60s) settled quota, paid scaling off,
actual counter/window and conservative observed charging rate. Worker time is not
charged quota. Initial reservations total12,000s; the sole optional compatible
resume has worker timeout≤2,100s and overhead300s, reservation≤2,400s. Reservations
and observed charges must stay within14,400s; no fourth session or paid fallback.
Exactly one requested/observed Tesla T4 is required; no CPU/multi-GPU training
substitute. Stop safely if accounting/resource availability cannot cover the next
session. No new GPU smoke job is permitted.

A controller lock, durable pre-push receipt/reservation and exact remote inventory
prevent duplicate attempts. Unique kernel names are used once, version1 only.
Provider ambiguity or process interruption never triggers a blind replacement.
The read-only reconcile command validates the recorded unique job's pulled source
hash/private GPU metadata before adopting its existing version. Source/config/data
or quota-window drift blocks continuation. Poll no faster than180 seconds. Every
completed/paused artifact is nonce/source/report/output-hash verified in private
scratch; failed and partial outputs are preserved. A failed/incomplete matched
pair cannot produce a new winner or automatic model promotion. No monitoring is
promised after a task ends.

## BRSET and independent evaluation remain a separate gate

BRSET's public dictionary and current1.0.2 access requirements may be checked
without accessing patient files. Kaggle credentials do not grant PhysioNet access.
The user must obtain [PhysioNet credentialing](https://physionet.org/settings/credentialing/),
complete the [required CITI training](https://physionet.org/content/brazilian-ophthalmological/view-required-training/1.0.2/),
and personally [sign the BRSET DUA](https://physionet.org/sign-dua/brazilian-ophthalmological/1.0.2/).
No account, agreement or access request is submitted automatically. Do not pay for
access or upload restricted data to Kaggle without an approved lawful arrangement.
Actual BRSET pair counts, labels, license entitlement and content-level overlap
remain unverified until permitted metadata/files can be inspected. No automatic
ODIR label mapping is approved, especially H/O/D, C or G; unknown labels stay
unknown. No full-eight independent cohort has been established.

This implementation separates **technical training readiness** from independent
evaluation readiness. The two fits use ODIR development roles only. A future user
execution request must expressly bind the frozen identities/budget and acknowledge
that external confirmation is blocked; no clinical/full-eight generalization
claim or demo replacement follows. This is an explicit clarification of the
planning prerequisites, not silent approval of external data or relaxation of its
label/overlap gates. Calibration fitting, external inference and final evaluation
require their own subsequent scope and readiness checks. The existing shared-T
design and all-eight metric/grouped-interval/reporting protocol remain reserved.

## Commands

CPU-only local checks, requiring no new execution authorization:

```sh
python scripts/candidate_experiment.py status
python scripts/candidate_experiment.py check-quota
python scripts/candidate_recover.py
python -m unittest discover -s tests -p test_candidate_readiness.py -v
```

`candidate_recover.py` preserves existing private inputs and restores only missing
actual pinned TRAIN224/TRAIN320/validation224/calibration224/validation320/
calibration320 inputs on a fresh task. It creates no approval. Securely configure
the existing Kaggle SDK and use the locked CPU recovery environment; no token
belongs in Git, logs or shell command text.

Only after a later exact GPU execution approval is saved privately:

```sh
python scripts/candidate_experiment.py advance --submit --approval-file artifacts/candidate-v14/experiment/approval.json
```

This performs one bounded advancement, never an unbounded polling loop. Repeat
only after the prescribed interval, inspecting saved status/receipts first.
Without that approval it fails before API access. `retrieve` recovers existing
terminal outputs; `reconcile --run-id <recorded-run-id> --kernel-version
<recorded-owner/kernel/1>` is read-only and refuses unrelated versions. Freeze is
an already-completed, non-repeatable readiness action; use the saved registry.
Actual identities, source revisions, job/version receipts and readiness evidence
are in the permitted aggregate readiness records, with patient artifacts private.
