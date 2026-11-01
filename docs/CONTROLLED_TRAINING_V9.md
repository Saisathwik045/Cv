# Controlled second development experiment v9 — approval pending

This protocol is prospective for new jobs, written after a checksum-bound audit
of v5 histories and development predictions. PR #17 is already merged at d7d7187.
All five original factories and the frozen research demo stay unchanged. No new
training, GPU smoke, cache build, split generation or duplicate review is authorized
by this document. `--submit` requires a separate recorded user approval bound to
the exact plan/source hashes, four run IDs and budget.

## Diagnosis and reason for the bounded changes

The generated `results/round2-v9/development-audit.json` reproduces all five
selected-checkpoint validation metrics from verified private prediction files and
verifies original receipts/source/checkpoint/history/cache hashes. It includes
all-eight per-class metrics/support, curves, weights and measured timings.

ConvNeXt training image weighted BCE falls 0.9510 → 0.3713 over five epochs;
validation patient weighted BCE bottoms at epoch 2 (0.7934), then rises to 0.9864.
DenseNet training loss falls 0.9495 → 0.4477; validation loss bottoms at epoch 3
(0.8038), then rises to 1.0858. The objectives/units differ (image training vs
mean-eye patient validation), so their absolute difference is not a directly
comparable generalization-gap estimate. Their trends suggest overfitting or
probability overconfidence, not evidence that simply training longer will help.

There is also an observed selection mismatch: the historical protocol selects
minimum-loss checkpoints, while comparing backbones by AUROC. ConvNeXt's saved
trajectory reaches validation AUROC 0.83733 at epoch 4 versus selected epoch 2
0.82876; DenseNet reaches 0.81417 at epoch 4 versus selected epoch 3 0.81189.
These later epoch scores are already development exposure, not new frozen models
or unbiased improvements. Historical checkpoints/reports are never replaced.
V9 explicitly changes checkpoint/early-stop selection to the primary ranking
objective. The v5 loss-based rule remains valid historical behavior, not a bug.

No scheduler/augmentation was used; AdamW was fixed at 1e-4, 224 pixels and full
fine-tuning. A smooth decreasing learning rate is one controlled way to test
whether smaller later updates improve ranking before stopping. The new baseline
first establishes the longer ceiling/patience and AUROC-selection regime; it is
not a one-factor causal comparison against v5, which differs in selection/budget.
Within v9 each paired comparison changes **only the learning-rate schedule**.
The 25-epoch cap is a ceiling, not evidence of underfitting or a promised duration.

Unclipped training positive weights range 2.08–32.96. At fixed 0.5, ConvNeXt H has
16 positive validation patients, 183 predicted-positive, precision 0.0601 and
recall 0.6875; A/C/M have relatively high recall and lower precision. N/D/O ranking
is weaker (AUROCs about 0.736/0.731/0.662). This supports separating weighted-loss
probability/threshold effects from ranking limitations. It does not establish
annotation errors. Threshold tuning cannot repair ranking; longer training cannot
be presumed to repair threshold-dependent F1.

## Exact matrix and common rules

| Order / run ID | Backbone | LR schedule | Maximum epochs |
|---|---|---|---:|
| convnext_constant | ConvNeXt-Tiny | constant 1e-4 | 25 |
| convnext_cosine | ConvNeXt-Tiny | cosine 1e-4 → 1e-6 over 25 epochs | 25 |
| densenet_constant | DenseNet121 | constant 1e-4 | 25 |
| densenet_cosine | DenseNet121 | cosine 1e-4 → 1e-6 over 25 epochs | 25 |

Every run starts from the same verified ImageNet V1 initialization for its
backbone, not v5/pilot/demo checkpoints. Common seed 2026 binds Python, NumPy,
Torch, CUDA and checkpointable loader RNG. One seed is a controlled initial
comparison, not an estimate of training-seed variance; no seed search is allowed.
FP32, AdamW, weight decay .01, batch 16, no augmentation, original EXIF RGB224
letterbox/ImageNet normalization, eight independent logits, trained dropout head
p=.2 and weighted BCE are unchanged. Weights are neg/pos **training patients only**.
Both eyes and connected containment groups retain their immutable v4 roles.
Patient probabilities remain mean(left sigmoid, right sigmoid), before thresholds.

Checkpoint = maximum all-eight patient validation macro AUROC, earliest exact tie.
Early stop = six completed epochs without >1e-4 AUROC improvement relative to the
last meaningful reference; any strictly higher score still updates best weights.
Undefined/nonfinite AUROC blocks the run; no class is dropped. Scheduler steps
once after a completed training/validation epoch; checkpoints save its next-epoch
state and optimizer LR. Both best and resumable last checkpoints are written every
epoch. Patient weighted BCE and actual LR remain diagnostics, not selection rules.

Compare all four best-AUROC checkpoints only after all four complete. Primary =
patient macro AUROC; exact ties retain matrix order. Report image and patient
macro/micro F1, macro/per-class AP, all-eight AUROC/precision/recall/F1 and positive,
negative/predicted-positive support at p>=.5; correct (n,8) formulas, no argmax.
Report labelwise binary accuracy, exact match and Hamming distinctly. Per-epoch
patient metrics are saved. Do not choose a winner from incomplete/undefined runs
or suppress unsuccessful results. No calibration fitting or risk-policy fitting.

## Development adequacy and protection of completed evaluation

Use exactly the existing 2,275 training / 525 validation patients (2,240 / 518
components; largest components 8 / 2). Every class has both outcomes. Validation
positive counts N,D,G,C,A,H,M,O = 175,169,32,32,25,16,26,147. Full positive/negative
patient and supplying-component counts are in the generated audit. Mixed-label
components retain their distinct labels; support counts are not independence or
power guarantees. Patient/component disjointness and both eyes were verified from
existing checked cache metadata, without recreating the split or inspecting pixels.

This validation cohort already selected five models and supported reliability
exploration. It can support this **small fixed development comparison**, with
optimistic adaptive-selection estimates and unstable rare-class effects. It is
not a fresh test set, and repeated epoch comparisons add selection optimism.
A new independent suitable cohort, preserved patient/identity boundaries, adequate
all-eight support and a new prospective evaluation protocol are required for a
new generalization/performance claim. Do not call these results final or clinical.
No existing role is reassigned. Additional search/seed sweeps are not automatic.

The old final holdout has been exposed; its predictions/reports/hashes are retained
as historical artifacts. The preservation audit reads aggregate endpoint records
only to verify the stated history, and hashes private NPZ bytes **without loading
labels/probabilities**. That separate preservation record is not a queue or tuning
input. It is never used for run/epoch/threshold/strategy/reporting-case selection.
No final or calibration images, labels, caches, predictions, manifests or raw ODIR
dataset sources are mounted in new GPU jobs. Jobs attach only the old train/val
cache (including ImageNet states) and, when resuming, their own preceding output.
The old calibration set/calibrator remain tied to the old candidate; they cannot
be silently reused or refitted for new weights. Demo remains REVIEW REQUIRED.

## Separate future experiments, outside this approval

Higher resolution (one 320-pixel arm) and one weaker-imbalance alternative
`BCEWithLogitsLoss(pos_weight=sqrt(training neg/pos))` are **deferred**, disabled in
this queue. The observed high rare-label positive bias motivates considering
weaker weighting; 224-pixel downsampling is a possible detail limitation, not a
proven bottleneck. Neither arm is assumed helpful. A later baseline review needs
a new fixed paired design/budget/approval before new caching or fitting.

Per-class threshold fitting is a separate study, also disabled here. Prospectively
maximize macro F1 using per-class F1 on a fixed .05:.05:.95 grid; exact ties choose
nearest .5 then higher. Fit only from training-only out-of-fold predictions with
containment-group-aware folds and both eyes kept together. Require >=10 positive
and >=10 negative supplying components per fitted label, otherwise retain .5.
The implementation requires explicit `train_oof` role/permission and is tested
only synthetically; no current saved predictions are used to tune thresholds.
Training-only grouped OOF fits and fold-support audit will require a separate
approval/budget; no training folds are generated in this task. Validation after
model selection would still be descriptive, not an unbiased threshold assessment.
Calibrator fitting remains a different authorized phase, never on the old final.

## Compute estimate, ceiling and safe continuation

Observed whole-worker times: ConvNeXt 541.562 s / 5 epochs; DenseNet 376.407 s /
6 epochs. Linear 25-epoch estimates are 0.752 h and 0.436 h per run: **2.376 worker
hours for four runs**. These include historical worker overhead but do not equal
exact GPU-quota billing. Early stopping can reduce time; cold provisioning,
checkpoint writes and resumes can increase it. No speed or accuracy improvement
is promised. Current read-only quota observation: six free hours, zero used or
reserved, paid scaling false; availability must be rechecked before each job.

Request **at most four GPU-quota hours**: at most six half-hour job submissions
(three hours hard worker timeouts), conservatively charging an extra ten minutes
per attempt for setup/teardown, to reserve four hours total. Max two attempts per
run. A failed/ambiguous attempt also consumes its full reservation; no automatic
refund or free retry. Thus interruptions can leave the matrix incomplete and
require a new approval. Each next job must fit both conservative reservations and
same-window observed quota consumption plus its guard. API accounting may lag,
and provisioning is outside worker control: these bounds/guard are not a promise
of second-exact billing. Stop on quota exhaustion, budget/attempt cap or unresolved
failure; no paid scaling or automatic budget extension is permitted.

Worker sessions stop at epoch boundaries with a 180-second finalization margin;
continuation restores strict model/optimizer/scheduler and Python/NumPy/Torch/CUDA/
loader RNG state from a compatible last checkpoint, with all provenance/hashes
checked. No historical model checkpoint may initialize a new fit. Completed runs
are never repeated. Existing job IDs/versions/source/nonce and remote inventory
are reconciled before every submission. Unknown/ambiguous jobs block submission;
recover their exact version/source receipt rather than create a replacement.
Results, best/last checkpoints, predictions, LR/timing histories and unsuccessful
attempts stay durably in original private Kaggle versions; local mirrors/receipts
stay outside Git. Aggregate queue state and permitted summaries are versioned.

## Commands and the execution boundary

```sh
# After configuration/code/protocol are committed; CPU/read-only preparation:
python scripts/diagnose_round2.py
python scripts/controlled_experiment.py prepare
python scripts/controlled_experiment.py check-quota
python scripts/controlled_experiment.py status
```

`prepare` refuses to overwrite an existing queue; `restore` recovers the saved
aggregate queue/receipts on a fresh task. No training job is submitted by these
commands. The single approval requested is the exact four-run matrix and budget.
After that user message, record a private approval JSON with `user_authorized=true`,
`plan_sha256`, `experiment_source_sha256`, exact `run_ids`,
`maximum_gpu_seconds=14400` and the user-message `approval_reference`. Then:

```sh
python scripts/controlled_experiment.py advance --submit --approval-file <private-approval.json>
```

One invocation performs bounded reconciliation and at most one submission. Save
state and return rather than promising background monitoring. Without --submit,
advance reconciles known jobs only. With missing/mismatched approval, submission
fails before an API client or external mutation. Keep the plan/source unchanged;
compatibility changes require review/new binding. After all four complete,
`compare` reads verified results only. No candidate replaces the frozen demo in
this phase; new performance claims await fresh independent evaluation.
