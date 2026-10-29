# Five-backbone development comparison v5

The user authorized five sequential free-quota Kaggle training runs after the
verified v4 pilot. PR #11 is already merged (`3ac4032`), with passing final CI.
The full experiment starts from ImageNet V1, **never the subset pilot checkpoint**.
Architecture, eight independent logits, dropout head, all five backbones, and
mean left/right sigmoid patient aggregation remain unchanged.

`configs/full-backbones-v5.toml` is the common prespecified configuration. All
models receive the same frozen v4 train/validation rows and 224-square EXIF/RGB
letterbox preprocessing with ImageNet normalization. No augmentation is applied.
Raw images are immutable. A single CPU cache job verifies the existing frozen
training/validation image hashes and applies the existing shared preprocessing;
it does not rerun image/duplicate validation or generate partitions. Only its
private train/validation pixel cache, metadata and hash-verified official ImageNet
weights are mounted in GPU jobs. Calibration/final data sources are absent.

## Training and selection

Fixed seed 2026, FP32 AdamW (learning rate 0.0001, weight decay 0.01), batch 16,
maximum ten completed epochs, no scheduler. Positive BCE weights are the
training-patient negative/positive ratios, without clipping. Each patient has
both eyes; these ratios also equal training-image ratios. Validation contributes
no training weights. Full fine-tuning updates backbone and eight-logit head.

Checkpoint selection uses **minimum validation patient weighted BCE**, retaining
the earliest epoch on an exact tie. Patient probabilities are the arithmetic
mean of both eye sigmoid outputs; patient loss is the mean weighted binary
cross-entropy of these probabilities, clipped to [1e-7,1-1e-7] for logarithms.
Training loss remains image-level BCEWithLogitsLoss. Early stopping uses the same
patient validation loss: patience three epochs without an improvement exceeding
0.0001 relative to the last meaningful improvement. Smaller improvements can
update the best checkpoint while the patience counter continues. Both rules are
implemented and tested separately. Selection never reads final evaluation.

The backbone comparison uses the best-loss checkpoint of each model. Primary
criterion: **validation patient macro AUROC across all eight labels**. All five
runs must complete and all eight AUROCs must be defined before declaring a
winner. Exact ties retain the fixed configuration order. This is development
selection, not unbiased final performance. No seed search, hyperparameter search,
threshold optimization, temperature fit or acceptance-risk fit occurs here.

## Metrics and uncertainty

Initial threshold is p >= 0.5. Report image and patient per-class AUROC, AUPRC
(average precision), precision, recall, F1, positive/negative and predicted-positive
support; macro/micro F1 are computed on (n,8) arrays. Report macro AUROC and average
precision, subset exact-match accuracy (all eight decisions correct), labelwise
binary accuracy (fraction of all binary decisions correct), and Hamming loss.
Ranking metrics require both outcomes; undefined values are null. Undefined
precision/recall/F1 denominators use zero, with supports visible. No clinical
performance, calibration or risk guarantee follows from this comparison.
Validation H has only 16 positive patients; class-specific estimates are unstable.

Inference timing is a warmed validation pass including cached input normalization,
transfer, forward, sigmoid and CPU collection, excluding metrics and original
JPEG decode/resize. Two warm-up batches are excluded. Report batch size, GPU,
parameter count, epoch times and whole worker elapsed time. Whole worker elapsed
is not an exact accelerator billing measurement; separately preserve quota API
observations and session reservations where available.

## Durable execution and recovery

`scripts/full_experiment.py` provides init, restore, retrieve, status, advance and bounded drive. It has an exclusive
controller lock, atomic registry writes, write-ahead submission receipts and
source/config/input identity checks. Each advance performs a bounded status check;
`--submit` may submit one next authorized run after reconciliation. It never
repeats a completed run. Ambiguous submissions block automatic retries. Unknown
remote full-v5 jobs also block submission until recovered. No paid fallback.

Every epoch atomically writes best/last checkpoints, history and progress with
checksums, then a checkpointed report. Resume restores model, optimizer, Python,
NumPy, Torch, CUDA and explicit loader-generator RNG at a completed epoch boundary.
Resume requires identical source, initialization, split, preprocessing, config
and compatible Python/Torch/NumPy versions. The previous private job output is
attached and progress/checkpoint hashes are checked. Completed epochs are not
retrained. Failed jobs require evidence review before their state can be changed;
no automatic failure retry is implemented. A paused valid job can resume.

GPU jobs are private and pinned by job/version. Their outputs persist on Kaggle;
large checkpoints, sample-aligned predictions and logs stay outside Git. Retrieved
source/privacy/resource settings, nonce, source revision, report checksum and
**all** artifact checksums are verified. The private run registry and index retain
exact paths and provenance; permitted aggregate registry snapshots expose no
patient IDs or sample-level values. A fresh task must reconcile the recorded
remote versions, not create replacement jobs. See STATUS.md for current commands.

The next phase, after the five-model validation comparison completes, is
reliability and shared-temperature calibration experiments under the frozen
protocol. Final evaluation stays untouched throughout model/result selection.

For a fresh task, restore the aggregate snapshot, then use `retrieve` to recover
completed models before comparison. The next `advance --submit` reconciles any
active job and resumes a paused compatible job. `drive --submit --max-seconds 1800
--poll-seconds 60` bounds automatic queue advancement to 30 minutes; it saves
state and returns at the budget boundary or on a blocker. No detached service is
installed. Use `requirements/full-v5-reports.lock` for the comparison environment.
