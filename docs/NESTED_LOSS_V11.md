# Training-only nested loss study v11 — execution approval pending

No GPU job, cache build, training or inference is authorized by this document.
The user requested preparation and a reviewable concrete budget. All five
backbone factories, frozen demo, historical evaluation, calibration and v4 roles
remain unchanged. Do not deploy v10 thresholds or transfer the old temperature.

## Evidence, hypothesis and one changed factor

V9 ConvNeXt cosine selected epoch 13, validation macro AUROC .84173, macro AP
.61131, macro F1 .59263. At .5, H TP/FP/FN were 3/1/13, with only 16 positive
patients. Validation weighted BCE increased after early epochs while ranking
selection favored a later epoch: this suggests overconfidence/overfitting rather
than evidence that more epochs alone fix rare-class performance. V10's
threshold-only CV raised macro F1 .59263 → .6055, but added 182 FP
to avoid 90 FN; C worsened and H stayed at 3/1/13. It did not establish an
independent improvement or a reliable clinical threshold.

Reference: fresh ImageNet ConvNeXt-Tiny, cosine schedule, BCE with positive
weight negative/positive count. Alternative: identical setup and BCE with positive
weight sqrt(negative/positive). Counts come **only from that fold's fitting
patients**, once per patient, never inner or outer labels. The reference's large
rare-label weights (historically up to ~33) can amplify noisy/sparse positive
examples; square-root weighting is one predeclared regularization hypothesis.
It may reduce excessive gradients or may hurt rare-class recall. Neither ranking
nor H performance is assumed to improve. Do not add focal-loss parameters,
resampling, augmentation, resolution, backbones, seeds or threshold searches.

## Nested assessment design and exact six-fit matrix

Use only the 2,275 existing v4 TRAIN patients / 2,240 containment components from
the pinned, already validated train/validation cache. Do not rebuild inputs or
read old validation/calibration/final rows/images/manifests. The cache contains
old validation files physically, but the new loader opens only its train entries.
Existing ImageNet-normalized preprocessing contains no cohort-fitted statistics.

Three outer containment-group folds; in each outer-development portion create
four group-aware inner folds, use inner fold 0 for checkpoint selection, and the
other three for fitting (approximately 50% / 16.7% / 33.3% of original train).
This is three-fold outer CV with **one fixed inner holdout**, not a costly inner
hyperparameter CV sweep. No refit on the complete outer-development portion.

Generalize the existing rarest-remaining-label allocator to K folds, preserving
whole components and using fixed support repair. Sort patient/component IDs;
seed 2026 for outer and inner allocations. Equal outer thirds and equal inner
quarters; fixed class/outcome/role ordering and tie breaks; no seed or partition
search. Both eyes share a patient; all linked patients share every applicable
partition. Mixed-label components retain individual labels. Save private fold
assignments/row indices and immutable hashes; old v4 manifests remain untouched.

Require each outer assessment to have >=10 positive-supplying components per
class; inner selection >=10; fitting >=10; all roles >=10 negative-supplying
components. These are feasibility floors, not power or clinical adequacy claims.
Report positive/negative patients AND components for all eight classes in every
role. If construction/support fails, stop without seed retries or dropping labels.
The 25% inner holdout reserves more positive support for checkpoint selection;
H remains sparse even with these larger assessments. The reduced fitting set
size changes the sample-size regime from v9; compare arms within this study,
not as a direct performance contrast against the full-data v9 model.

| Order | Outer fold | Arm |
|---|---:|---|
| 1 | 0 | ratio_bce reference |
| 2 | 0 | sqrt_ratio_bce |
| 3 | 1 | ratio_bce reference |
| 4 | 1 | sqrt_ratio_bce |
| 5 | 2 | ratio_bce reference |
| 6 | 2 | sqrt_ratio_bce |

Fresh official ImageNet V1 initialization per fit; never v9 or demo weights.
Both arms share seed 2026 and per-fold data/order. Preserve RGB/EXIF 224 letterbox,
ImageNet normalization, eight independent logits, trained dropout head .2,
FP32 AdamW lr 1e-4 / weight decay .01, batch 16, no augmentation, cosine 1e-4 →
1e-6 over maximum 25 epochs. Patient predictor remains mean of both eye sigmoid
outputs. No classwise threshold fitting: all results at p >= .5, uncalibrated.
One training seed is a controlled paired comparison, not seed-variance evidence.

## Checkpoint and comparison rules before execution

Checkpoint and early stopping use **inner-selection patient macro AUROC** only,
all eight defined; best strict maximum, earliest exact tie. Stop after six epochs
without improvement >1e-4 relative to the meaningful reference, ceiling 25.
Save best and resumable last each epoch, optimizer/scheduler and all RNG states.
Never evaluate outer images/predictions on paused runs or during epoch selection.
After completion strictly reload best once, then evaluate outer patients once.
All six settings remain fixed even if a preceding outer result is visible.

Primary paired loss contrast: mean across three folds of
(sqrt-ratio outer macro AUROC minus reference outer macro AUROC). Candidate arm
for future study: higher mean outer macro AUROC, exact tie reference. No selection
from incomplete matrix. This comparison is itself method development, not an
untouched final performance claim. Report per-fold and mean metrics, fold
variation, plus pooled patient-level outer predictions with explicit warning
that scores come from three differently trained models; pooled AUROC/AP are
secondary and can be distorted by fold score scales. Outer assessments are
mutually disjoint, but training fits overlap; do not call three fits independent
replications or use a naive fold t-test.

Report macro AUROC, macro AP, macro/micro F1, exact match, Hamming and labelwise
accuracy distinctly. All eight classes: AUROC/AP, precision/recall/F1,
positive/negative patients/components and TP/FP/FN/TN. Report H's per-fold support,
TP/FP/FN, recall and AP even when unfavorable. Undefined metrics remain null;
zero-denominator classification metrics retain flags. No class is silently removed.

Paired component-cluster bootstrap, 2,000 resamples, seed 2026, within each outer
fold, same sampled components for both arms; average the paired fold differences
per replicate. Percentile 95% intervals for macro AUROC/AP/F1 and H recall/F1;
undefined replicates are counted and reported, never silently discarded.
Percentiles use defined replicates with their count visible; if fewer than 90%
are defined the interval is null. CPU batches checkpoint every 20 replicates;
`report_nested_results.py --bootstrap-batch 100` resumes the same fixed 2,000
replicates without changing inputs or fitting anything. These
intervals condition on the fitted models and fixed study choices; they omit
training/selection variability and are exploratory, not clinical guarantees.

## Budget, queue and approval boundary

Propose a strict **14,400 seconds (four free Kaggle quota hours)** total. At most
six submitted sessions including any retry/resume, each hard timeout 2,100 s plus
300 s conservative reservation/overhead guard. No refunds for failed/ambiguous
sessions; at most two attempts per fit. The six planned fits leave no extra
session reserve: a pause/resume or failure can leave the matrix incomplete.
Preserve checkpoints and stop rather than exceeding the cap; any extension needs
new approval. No GPU smoke, full-data refit, paid fallback or follow-on fitting.

Before every future submission: reconcile receipts/exact remote versions and
inventory, inspect free quota and paid scaling false, guard next reservation
against max(conservative spent, observed same-window quota delta). Sequential
jobs only. Ambiguous state or unknown matching job blocks, never blindly resubmit.
Poll at most once per advance, no more often than 180 seconds; no background
monitoring promise. Never resubmit completed runs. Preserve successful, paused
and failed versions, logs, source/nonce receipts, checkpoints/predictions/checksums.
Approval must bind the actual plan/source hashes, six exact run IDs and 14,400 s.
Prepare/build/CPU tests are not training execution evidence.

Estimate from measured v9 ConvNeXt whole-worker seconds per completed epoch,
scaled separately by actual fitting and selection patient counts. Retain the
measured whole-worker minus epoch-compute overhead per epoch rather than scaling
checkpoint/IO overhead down with data size.
Add one outer pass and an explicit per-fit 180-second cold-start/IO allowance.
Publish numeric estimates from real fold counts and timings, including the
25-epoch ceiling and uncertainty; scaling is approximate, not a quota guarantee.
Current quota is observed read-only; availability must be rechecked after approval.

## Commands and recovery

Preparation is CPU-only: `python scripts/nested_experiment.py prepare`.
`status` reads the queue; `check-quota` is read-only and never submits.
Do not create an approval record until the user approves the exact plan/budget.
After approval, record `user_authorized`, `plan_sha256`,
`experiment_source_sha256`, `maximum_gpu_seconds: 14400`, all six `run_ids`
and the actual `approval_reference` in private `approval.json`.
Only then use `advance --submit --approval-file artifacts/runner/nested-v11/approval.json`.
Every advance reconciles existing jobs before a new submission.

A fresh task can use `restore` with the committed queue/receipts and original
train metadata. It regenerates only the deterministic private CV assignments,
requires the exact saved hash, and refuses to overwrite an existing registry.
`retrieve` downloads terminal exact-version outputs read-only and verifies
source/nonce/report/all output hashes. It never submits a job. Missing original
cache metadata must be retrieved from its pinned private version; never rebuild
the dataset/cache. Source/configuration changes block compatibility.
`python scripts/report_nested_results.py --bootstrap-batch 100` refuses a partial
six-fit matrix, then performs checkpointed CPU reporting only. No unfinished
command is advertised as completed evidence.

## Remaining independence limits and fresh confirmation

New fits exclude each outer patient's pixels/labels from weight updates,
checkpoint selection and loss weights. This addresses the v10 within-study
model-selection contamination structurally. It does **not** erase history: all
these training patients helped train prior models; reference, loss hypothesis,
architecture and study design reflect earlier development. Same-source ODIR,
patient-derived labels, sparse H and one seed constrain interpretation. The
old exposed final holdout is historical and must never confirm the new choice.

A fresh suitable labeled independent cohort must contain new patients/identities
and no duplicate/copy links to any development cohort; preferably a temporal or
external-site/device population representative of intended use. Preserve paired
left/right acquisition and prespecified patient aggregation. Obtain consent/data
rights and authoritative all-eight multilabel patient annotations under a frozen
labeling protocol, independently reviewed by qualified graders with blinded
conflict adjudication and explicit uncertain/missing-label handling. Do not infer
eye-specific lesion truth from patient labels. Use consecutive or prospectively sampled cases, not cases chosen for model
errors. Include realistic negatives and co-occurrences; enriched rare-label sampling needs explicit sampling weights and
cannot establish deployment prevalence/calibration/PPV unweighted.

Plan size from a prespecified precision or paired-comparison power calculation,
including component clustering, multiplicity and all-eight positive/negative
support. As a nonclinical precision illustration, worst-case binomial recall
half-width .10 at nominal 95% requires about 97 independent positive patients
per label before clustering/design inflation; this is not a clinical recall
target or a validated power calculation. H's tens of current cases are inadequate
for narrow confirmation. Actual cohort size depends on prevalence, desired
precision/effect and design; do not promise a target accuracy.

Freeze new weights, preprocessing, aggregation, disease thresholds and any
separately fitted calibration/reliability procedure before new evaluation access.
Do not tune on the fresh cohort. No new cohort collection or calibration fit is
executed/authorized here. Existing demo remains REVIEW REQUIRED.
