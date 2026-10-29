# Prospective development protocol v3

[Development v3](DEVELOPMENT_V3.md) and `configs/development-v3.toml` define the
current bounded verification and simpler shared-calibration design. This supersedes
the infeasible requirement for independent rare-class calibration subdivisions
for the development pilot only. No per-class calibration or fitted clinical risk
policy is claimed. All five backbones and architecture components remain.
Real-data readiness is conditional on a new audit; historical splits stay unapproved.

# Review v2 supersedes historical split acceptance

See [independent duplicate and feasibility review](DUPLICATE_REVIEW_V2.md).
Historical candidate-connected manifests are diagnostic sensitivity artifacts.
Confirmed/probable-copy links remain mandatory groups; weak candidates require
independent evidence or explicit quarantine, never automatic confirmation.
Before fitting, freeze validation-only selection, distinct calibration-fit and
calibration-policy groups with all-eight-class support/power, a prespecified risk
budget and confidence bounds, and an untouched final holdout. One H-positive
calibration patient cannot support class-specific fitting or disjoint supported
subdivisions. No revised cohort or experimental split is approved.

# Experimental readiness: blocked pending duplicate/protocol review

The rules below include historical Phase 1 choices. The blanket weak-candidate
co-grouping policy and its published manifests are sensitivity cases, not approved
experimental partitions. A dHash edge does not establish duplicate identity.
Do not train or fit calibration/reliability thresholds under the historical
one-H-positive calibration split. The versioned review will distinguish confirmed,
probable, unresolved and evidence-supported false-positive relationships using
independent image evidence; unresolved material risks need sensitivity/quarantine.
No threshold, seed or partition may be chosen using model performance.

# Experiment protocol (prespecified foundation)

## Dataset and partitions

Original ODIR-5K `data.xlsx` is authoritative. Phase 1 must inspect its actual schema;
expected fields include ID, Left-Fundus, Right-Fundus and N,D,G,C,A,H,M,O.
Never infer disease targets from filenames, diagnostic text or a repackaged CSV.
Validate unique patient IDs, exactly eight binary nonmissing labels, image existence,
decoding, eye association, duplicate references/content and workbook provenance.
Flag contradictory N/other labels and unresolved duplicates for explicit adjudication;
do not silently change labels or delete images. Quarantine decisions are versioned.

Split at patient level before expanding to eyes using deterministic multilabel
stratification. Initial fractions: train .65, validation .15, calibration .10,
final evaluation .10, seed 2026. Persist the exact patient assignments; fractions
are targets, not guaranteed counts. Require at least one positive and one negative
patient for every class in every partition, and report full counts (this minimum
is a feasibility floor, not evidence of statistical power). Abort when infeasible;
review a new protocol before training rather than retrying seeds until results look
favorable. Exact/near-duplicate images across patients must be resolved or grouped
before splitting to prevent leakage. Keep both eyes in the same group.

Official test images lacking authoritative labels cannot contribute to metrics.
Only labeled, eligible patients may enter the four partitions. Preserve any external
official partition boundaries and record the source cohort; never invent labels.

## Duplicate containment policy (version 1)

Use `conservative_patient_components_v1` in Phase 1. Preserve the existing
64-bit horizontal dHash detector and Hamming distance <=2, alongside raw-file and
EXIF-oriented decoded-RGB SHA256 equality. Enumerate all pairs rather than stopping
at the first. Exact equality confirms identical file/pixel content; a perceptual
candidate alone remains unresolved. Aligned thumbnail MAE/PSNR/correlation,
contrast/entropy and metadata provide descriptive technical evidence, never a
clinical decision or a model-performance-selected threshold.

Retain every image and patient-derived annotation. Co-group all candidate-linked
patient IDs, including transitive connections, into indivisible components. Both
eyes stay with their patient. This conservatively contains unresolved pairs and
possible hash false positives. Conflicting annotations are recorded without
relabeling or asserting that different patients are the same person. No candidate
is dismissed on visual inspection alone. Within-patient duplicates remain in the
same component. Report retained duplication and label conflicts as limitations.

`trustretina-component-rarest-label-repair-v3` assigns components using the fixed seed and
original fractions as patient-count targets. Whole groups can change realized
counts. No seed search occurs. Fail before publication if connected groups make
four-partition positive/negative support infeasible, or the deterministic assignment
cannot repair actual support. After initial allocation, reduce support deficits in
fixed partition/class/outcome order by moving whole components. Never create or
increase a donor deficit. Prefer the smallest component, then least total patient
count deviation from the fixed fraction targets, then the existing seeded rank.
Each move strictly reduces the integer deficit; no seed search or model performance
criterion is used. Record every move privately. If no safe move exists, fail closed;
this constructive repair does not prove all other assignments infeasible.
Independently recompute all candidate edges when auditing
manifests, check patient/content/component separation and immutable checksums.
Keep pair identities, evidence, decisions, reasons, annotation conflicts and exact
patient assignments in private, versioned adjudication/manifests outside Git.
Later calibration fit/policy subdivisions and uncertainty resampling must preserve
these same connected components, including both eyes and all linked patient IDs.

## Baseline comparison

All five configured models use eight independent logits with BCEWithLogitsLoss,
identical patient manifests, preprocessing, seeds, optimization budget and selection
rule. Initial common resolution 224, batch size 16, AdamW learning rate 1e-4,
30-epoch cap; freeze exact optimizer/scheduler/early-stop settings in Phase 2 before
running. Use ImageNet pretrained weights with recorded version/hash and matched
normalization; record all transforms. Apply stochastic augmentation only to training.
Any training class weights are computed from training patients only.

Provide the same explicit dropout head (p=.2) for every backbone so MC Dropout is
well defined even when the base network has no dropout. Select best weights by
validation macro F1 at fixed .5, then freeze the model and tune per-class disease
thresholds on validation only with a prespecified grid/tie rule. No final labels
may influence architecture, epoch, thresholds, hyperparameters or run selection.
Treat one configured seed as the initial baseline; add prespecified repeated seeds
later rather than claiming significance from one run.

## Metrics and reporting

Targets and binary predictions must have the same **two-dimensional (n_images,8)**
shape. Use `f1_score(y_true, y_pred, average='macro', zero_division=0)` on those arrays;
never flatten them or use argmax. Report micro/macro/per-class F1, precision, recall,
AUROC and average precision, plus support and patient counts. Mark metrics requiring
both outcomes undefined when support is missing, with the reason. Thresholds are
recorded with results. Use patient-cluster bootstrap intervals so both eyes are
resampled together. Persist raw logits/probabilities and labels with row identifiers
outside Git. Never populate comparisons manually; report missing or failed runs.

## Reliability and calibration

Confidence is a prespecified score over sigmoid outputs, not a softmax probability.
Define the acceptance score precisely in Phase 3 (candidate: minimum per-class
max(p,1-p)); report class imbalance limitations. MC Dropout runs 30 stochastic
passes with dropout enabled and BatchNorm/every other stateful layer in eval mode.
Persist predictive mean, per-class variance and entropy; record RNG seeds and cost.
Quality measures (e.g. blur, illumination, usable field of view) are proxies, not
validated clinical gradability. Fit quality normalizers on training data only.

Fit a multilabel calibration method, initially per-class temperature scaling of
logits, on calibration patients only. Prespecify an internal calibration fit/policy
subdivision or cross-fitting in Phase 3 to avoid fitting and claiming performance on
the same examples; check class support again. Disease threshold numerical values
are selected on validation, including how calibrated outputs are thresholded.
Select triage cutoffs on calibration-policy data under a prespecified coverage/risk
budget. Freeze calibrator, thresholds and triage policy before final evaluation.

Compare confidence alone, +uncertainty, +quality, both, and calibrated combined.
Include uncalibrated/calibrated paired controls to isolate calibration effects.
Report Brier score, multilabel reliability diagrams and explicitly defined ECE,
plus risk/coverage, accepted/review counts, sensitivity and false-negative rates.
Threshold definitions and zero-coverage behavior must be explicit. Allow all
methods to perform worse; no post-hoc exclusions or tuning on final labels.

## Decision and explanation

Triage follows reliability analysis and calibration. ACCEPT means the frozen
research policy accepted the prediction, not that the retina is healthy. REVIEW
REQUIRED includes reasons and abstains if quality or pipeline validity fails.
Grad-CAM is class-specific on the frozen model after the decision. Record target
layer/class and preprocessing geometry, preserve image alignment and test map
sanity. A heatmap is not proof of diagnostic correctness or clinical validation.
