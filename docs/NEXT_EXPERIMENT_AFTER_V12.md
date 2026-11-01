# One next experiment: input-resolution ablation (proposal only)

**Do not execute.** This document proposes a single factor, not an approved job
queue. No new cache, GPU session, model inference or upload is created in v12.
The original demo and both ratio/sqrt-ratio loss implementations remain intact.

## Why this factor, and what it cannot establish

The audit found substantial geometry compression: the model receives 224-square
letterboxed inputs whose median resized short side is only 150 pixels. Some
reviewed H true positives and false positives share small retinal fields and
wide borders. H false negatives are not all visibly obscured, and their scores
are often far below .5. O errors also occur with readable major structures.
These observations justify testing preservation of spatial detail, not claiming
that insufficient resolution causes errors or that more pixels will help.

Increasing resolution preserves the source field instead of making an unverified
retinal crop. Do not combine it with cropping, augmentation, oversampling, new
losses, thresholds or label changes. Patient-derived targets remain an unresolved
limitation; this experiment will not establish eye-specific diagnoses or resolve
O's heterogeneous definition. Source-format shortcuts could persist at either
resolution. Clinical/schema clarification is needed independently of this test.

## Prerequisites before any training approval

1. Recover the original TRAIN-only images from existing pinned private storage
   in an authorized follow-up. Check their raw SHA256 against frozen rows and
   verify reproducible 224-cache parity on a prospectively fixed train-only sample.
   Only the 224 cache is local now: upscaling that cache would not test resolution.
2. Obtain source-annotation schema clarification from the data documentation or
   qualified data steward: H retinal label versus systemic hypertension; O's
   catch-all inclusion rules; patient versus eye localization and normal-eye
   keyword semantics. Do not silently relabel or adjudicate from these images.
   If required information remains unavailable, retain that limitation and decide
   whether the research question is defensible before spending GPU quota.
3. Freeze new source/configuration/cache hashes, exact run IDs, numerical loss
   and resume tests, predicted accounting and explicit user budget approval.
   Recheck actual free quota/accelerator charging before each submitted session.
   Unused v11 budget is not authorization for this study.

## Prespecified matrix and method

Six fits: unchanged v11 training-only containment folds 0,1,2 × **224 vs 320**
pixels. Each fold keeps exactly its saved fit/inner/outer patients and both eyes;
no reseeding/partition search and no old validation/calibration/final access.
Both arms start fresh official ImageNet ConvNeXt-Tiny with seed 2026. Keep original
letterbox/EXIF/RGB, ImageNet channel normalization, eight independent logits,
per-eye ratio-weighted BCE using fit patients only, cosine AdamW lr1e-4→1e-6,
weight decay .01, max 25 epochs, no augmentation. Keep all five factories intact.

Use common microbatch eight images and accumulation two (effective 16), identical
optimizer-step cadence and remainder handling in both arms. Do not reuse the v11
224 checkpoint as the reference: rerun it with the same new implementation so only
resolution differs. Prespecify/check trainable-head and normalization behavior,
FP32 precision, exact accumulation scaling and scheduler steps before approval.
Save best and resumable last/optimizer/scheduler/all RNG every epoch.

Selection: highest **inner patient macro AUROC**, earliest exact tie; unchanged
six-epoch patience for >1e-4 meaningful improvement. Evaluate an outer fold once
after its best checkpoint is frozen/strictly reloaded. Patient predictor remains
mean two-eye sigmoid outputs; fixed p≥.5. No calibration or threshold fitting.

Primary contrast: paired mean outer macro AUROC (320 minus 224), with the existing
fixed-seed 2,000 paired component-cluster bootstrap and its conditional limitations.
Report macro AP, macro/micro F1, all-eight precision/recall/AP/FP/FN/support and
foldwise H; include co-label and technical-format summaries as descriptive checks.
A numerical mean lead alone does not establish superiority. If the AUROC interval
includes zero, report no established primary improvement; if F1/rare-class outcomes
worsen, explicitly reject a blanket performance-improvement claim. Do not change
thresholds to rescue a negative result. No winner from an incomplete matrix, no
single-fold global checkpoint promotion, all-data refit, calibration or demo update.
Prior v9–v12 choices still limit independence even with untouched outer assessment
within each fit. A fresh independent cohort is necessary for confirmation.

## Estimated cost and strict stopping rule

The v11 all-six, 25-epoch worker ceiling was 2.783 hours. Using spatial-area scaling
(320/224)^2≈2.041, a three-pair 224/320 ceiling is about
2.783×(1+2.041)/2 = **4.23 worker-hours**, before hardware/IO/accumulation effects.
A 30% planning allowance gives **5.50 hours**; this is not a charged-quota promise.
Actual v11 early stopping used 1.143 quota hours, but do not budget assuming that
will recur. Source-image decode and new CPU cache preparation are separate CPU
costs and must not be run as a GPU session.

Propose **six free GPU-quota hours (21,600 seconds), maximum six sessions**,
including retries/resumes/overhead. Per pair reserve 2,700 quota seconds for 224
(worker timeout 2,400 + 300 guard) and 4,500 for 320 (4,200 + 300). Three pairs
exhaust the reservation: there is no seventh session. Adjust only downward or
stop if measured accelerator billing cannot fit the next reservation. Budget
must be newly approved; no paid fallback. Any interruption/failure can leave the
matrix incomplete. Preserve failed attempts/checkpoints and stop, rather than
claim a resolution result or silently extend the study.
