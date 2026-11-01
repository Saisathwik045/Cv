# Prospective CPU-only threshold analysis v10

This document and configuration are committed before computing results. The user
has separately authorized exploratory threshold cross-validation on existing
validation predictions. This supersedes the **data-role restriction only** of
the deferred training-OOF threshold study in CONTROLLED_TRAINING_V9.md for this
analysis; that training-OOF helper remains unchanged and cannot be passed
validation data disguised as training. No new model fitting or inference occurs.

## Fixed inputs and estimand

Use only the saved deterministic raw mean-eye sigmoid probabilities for the
selected ConvNeXt cosine epoch-13 checkpoint from
`saisathwikregula/trustretina-round2-v9-convnext-cosine-ede050b3/1`.
Checkpoint SHA256:
`af33eb06fb2de21929e4dd3ec3959c0fbc58890d126a50c2e0d2b34bafaa0d49`.
Verify source/nonce/report/output checksums, the recorded prediction digest and
class order N,D,G,C,A,H,M,O, and reproduce the saved fixed-.5 baseline before
analysis. Verify mean-eye aggregation from saved validation image predictions.
All 525 validation patients / 518 containment components are retained. No final
or calibration images, labels, predictions or manifests are opened. Do not load
or apply the old calibration temperature to these new weights. The frozen demo
and historical reports are not changed; REVIEW REQUIRED remains the default.

Primary comparison: pooled patient-weighted out-of-fold per-class threshold F1
versus fixed p >= .5 on exactly the same patients. Predictions are positive on
p >= threshold, including equality. Compute precision TP/(TP+FP), recall
TP/(TP+FN), F1 2TP/(2TP+FP+FN), and TP/FP/FN/TN for each label. Zero denominators
use zero and retain defined flags. Macro F1 averages eight class F1s; micro F1
pools counts over the equal-shaped (n,8) arrays. Also report exact match and
labelwise binary accuracy separately. Threshold changes cannot change AUROC/AP;
no ranking improvement is claimed.

## Four fixed group-aware development folds

Build folds from patient IDs, binary labels and the saved containment-component
IDs only; do not expose model probabilities to the allocator. Sort patients and
components by string ID. Reuse the existing deterministic
`trustretina-component-rarest-label-repair-v3` allocator with four fractions .25,
seed 2026 and positive/negative patient support floors 1. Its rarest-remaining-label
allocation and whole-component support repair are fixed; no seed/method/fold
search based on predictions or results. The allocator's historical four role
names are used internally only: these are four **validation CV folds**, never
new train/calibration/final partitions. No frozen v4 manifest is overwritten.

For each fold, fitting uses the other three folds; its threshold assessment uses
only that fold. Every patient is assessed exactly once. All patients sharing a
containment component remain in the same fold; each patient already represents
both eyes. Audit label-positive/negative patient and supplying-component support
for fitting and assessment. Mixed-label components retain their patient labels
and may supply both outcomes. If fold support cannot be established, stop rather
than silently change the method or seed.

## Objective, candidate grid, ties and sparse support

For each class separately, maximize fitting-fold patient F1 on the fixed grid
.05,.10,...,.95. The independent per-class maxima maximize macro F1 on fitting
rows. Compare F1 ratios exactly with integer cross-products. Exact ties choose
nearest .5 using integer grid distance, then the higher threshold. No clinical
recall constraint or asymmetric error cost is invented. No objective/grid search.

Require at least 10 positive-supplying and 10 negative-supplying containment
components in each fitting fold for that label. Otherwise retain .5 and record
the fallback and support. Held-out labels never choose thresholds, eligibility,
ties or fallback. Do not optimize thresholds jointly on pooled held-out results.
Do not refit/export a single deployable all-data threshold vector after CV.

## Reports and stability

Publish pooled baseline versus held-out precision/recall/F1/counts for all eight
classes, signed TP/FP/FN/F1 differences, and each fold's assessment results and
support. Show the four fitted thresholds, median, quartiles (NumPy linear method),
IQR, min/max and count of distinct values. Four overlapping fits are not four
independent experiments; descriptive stability ranges are not confidence
intervals. Give H a separate table of fitting support, threshold/fallback,
held-out support and outcomes. Include a plot of paired FP/FN counts and threshold
variation. Preserve worsening results and explain whether fewer FP were bought
with more FN or vice versa. No automatic acceptance policy is fitted.

## Limitations, budget and preservation

This is conditional cross-validation of a **threshold-fitting procedure on fixed
predictions**, not cross-validation of model training or checkpoint selection.
All validation labels previously influenced epoch/backbone selection and other
development; held-out threshold folds are therefore not wholly independent of
the model-selection process. This procedure cannot remove that selection bias.
Rare H has only 16 positives overall; per-fold recall is highly discrete. No
clinical targets, safety guarantees or independent confirmation are supported.
Do not choose another threshold strategy from these results in this task.

One fixed four-fold analysis, CPU only, no GPU jobs, new predictions, calibration
fitting, retraining or paid resources. Sample-aligned fold assignments, decisions
and exact input/output hashes remain private under artifacts/analysis/threshold-v10.
Commit only code, protocol/configuration and aggregate outputs. Refuse to overwrite
a completed analysis with incompatible inputs; preserve failed attempts. Record
runtime, dependencies and source revision. Frozen demo files may be checked as
opaque bytes; historical final files are not opened even for a new hash audit.

Recommendation will be based on these descriptive findings, with a fresh suitable
independent cohort and a separately prospective frozen protocol required for any
new performance claim. No demo replacement is authorized.
