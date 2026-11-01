# Bounded development audit v12 — prospective rules

This CPU-only audit consolidates v9–v11. No training, new model inference,
threshold fitting, calibration, final-evaluation analysis, data upload or change
to labels/images/demo is authorized. Both loss implementations are preserved.
Budget: at most 30 minutes wall time for automated local CPU processing, one
pass over 4,550 cached training images, six saved outer prediction pairs. Visual
review is capped at 32 selected paired-eye records; do not expand after inspection.

## Data and provenance

Use v11 frozen TRAIN-only outer predictions from all six completed runs, verifying
input prediction and row/cache hashes against saved verified receipts. Validate
mean-eye probabilities, patient labels and containment IDs. Use v9/v10 aggregate
reports as historical development context, not new independent comparisons.
Do not open historical final/calibration prediction/image files. Filter shared
source annotation metadata and source-label inventory to the authorized 2,275
training patient IDs before analysis. Record incomplete metadata coverage and
selection bias (the keyword cache originated in duplicate-candidate review).
No diagnostic keyword maps may replace authoritative eight-label targets.

Available images are the pinned uint8 224×224 cached model inputs. No original
raw dataset is local; do not download/reconstruct it. Record that native image
quality, small anatomical details and raw-to-cache pixel parity cannot be judged.
Confirm recorded raw hashes/geometry contracts without claiming raw revalidation.

## Fixed sampling before image inspection

For each arm (ratio_bce, sqrt_ratio_bce), each label (H,O), and each outcome
(TP,FP,FN,TN) at p≥0.5, sort eligible outer patients by
SHA256(UTF8("trustretina-audit-v12|2026|" + patient_id)); choose the first two
from distinct containment components. Selection uses neither image appearance,
quality nor score extremity. No replacement for missing images or empty strata.
Patients may repeat across strata; retain membership, render each selected pair
once. Cap 32 stratum records / 64 eye slots. Show both cached eyes at identical
scale; private opaque case codes instead of patient IDs. Record technical review
for every selected pair, including correct predictions. Selected-case frequencies
are not estimates of population image-quality prevalence.

## Prespecified descriptive analyses

- H/O TP,FP,FN,TN and score quartiles by stratum, both arms; co-occurrence counts
  for every class; H/O errors split by single versus multiple positive labels.
- Mean-eye versus per-eye score disagreement and positive patients with one eye
  ≥.5 but mean <.5 (descriptive, not alternative predictor/threshold selection).
- Source-label/row consistency, N+disease and repeated-eye target checks. Keyword
  symmetry, literal 'normal fundus' in exactly one eye and literal 'hypertensive
  retinopathy' mentions for H. O keyword diversity is lexical, not adjudication.
- All cached images: nonpadding dark fraction, contrast and adjacent gradient
  energy inside non-dark foreground, exact mean-color fraction. No fitted quality
  normalizer, clinical gradability cutoff or significance fishing. Summarize pair
  means by outcome; signals are technical/descriptive and confounded by anatomy,
  device and pathology. Review padding/resize/normalization code and target loss.
- Visual fields: apparent sharpness, exposure/obscuration, retinal framing,
  padding and between-eye appearance at 224 pixels; categorical 'uncertain' allowed.
  No diagnosis, lesion localization, annotation-error claim or relabeling.

Keep IDs, keywords per patient, arrays and montages private under artifacts/audit-v12.
Publish aggregates, hashes, technical observations and limitations only. Distinguish
observed counts from causal hypotheses. Sparse support alone cannot explain H.
Prepare one prospective next-experiment recommendation with cost/stopping rule;
no runnable GPU queue or submission. Assess independent-cohort label compatibility,
representative sampling, privacy/rights, overlap checking and all-eight support;
no new cohort may be described as available without evidence.
