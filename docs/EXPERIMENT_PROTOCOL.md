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

`trustretina-component-rarest-label-v2` assigns components using the fixed seed and
original fractions as patient-count targets. Whole groups can change realized
counts. No seed search occurs. Fail before publication if connected groups make
four-partition positive/negative support infeasible, or the deterministic assignment
fails actual support. Independently recompute all candidate edges when auditing
manifests, check patient/content/component separation and immutable checksums.
Keep pair identities, evidence, decisions, reasons, annotation conflicts and exact
patient assignments in private, versioned adjudication/manifests outside Git.

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
