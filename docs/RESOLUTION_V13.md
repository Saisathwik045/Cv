# Matched 224-versus-320 experiment v13 — preparation only

No training is authorized by this document. A new explicit approval must bind the
final plan SHA256, worker/source SHA256, six run IDs and strict six-hour free-quota
cap. The six v11 runs, v12 audit and original demo remain historical/unchanged.

## Source semantics and research scope

The original [ODIR dataset description](https://odir2019.grand-challenge.org/dataset/)
and [challenge background](https://odir2019.grand-challenge.org/background/) identify
H as hypertension and O as other diseases/abnormalities. Labels apply to patients
using both-eye images, with age also available to annotators; multiple labels are
allowed. This does not define an eye-specific target vector. The dataset creators'
[OIA-ODIR description](https://github.com/nkicsl/OIA-ODIR) corroborates this scope.

These public descriptions do not specify an exhaustive O taxonomy, a keyword-to-
label algorithm, H severity thresholds or a systemic-diagnosis versus retinal-sign
case definition. The source's vascular explanation is not an individual clinical
adjudication. Do not rename H to a verified hypertensive-retinopathy diagnosis or
infer systemic disease from an image. Keep original eight-label workbook targets;
no relabeling or new clinical claims. This is a defensible ablation of **fixed ODIR
benchmark targets**, not a claim of clinically validated hypertension detection.
Unresolved semantics remain a blocker to clinical interpretation/independent
cohort label harmonization, not silently resolved by increasing resolution.

Direct official-page opens returned403; indexed official-page text and the
creators' accessible repository were reviewed, alongside the authorized Kaggle
mirror metadata. Source-access limitations and content hashes are recorded in the
preparation report. The mirror states source licensing is unspecified; this task
continues existing authorized private research access, with no public data release
or broadened rights claim.

## Native inputs and frozen folds

Recover only the 4,550 images of the frozen 2,275 TRAIN patients from Kaggle dataset
611716 version2. Verify every raw SHA256 against frozen train rows. Do not parse
calibration/final records, regenerate splits or revisit duplicate decisions.
A private GPU-disabled CPU job may mount the existing original dataset but opens
only paths in the frozen training allowlist; it reads only the old TRAIN tensor.
Persist its ID before submission and recover it rather than resubmit ambiguity.

Rebuild RGB/EXIF mean-color letterbox at224 directly from native images and require
**zero differing uint8 channel values on all4,550 images** against the pinned cache.
Any mismatch blocks preparation and training pending investigation. Build320 from
the same native decode with the same bilinear geometry and ImageNet normalization,
never by scaling the224 tensor. Record per-image raw/content hashes privately,
geometry, full cache/row/index/initialization hashes and source/dependency versions.
Original images, labels and old cache remain immutable.

Reuse the v11 manifest exactly (canonical SHA256
`e8f3e7a6e44880c2a5d09dac08619df029d650658d720b26b15352721ca844a5`).
Keep its three outer folds and each fixed inner-selection holdout, both eyes and
all containment-linked patients together. Support is not improved by repartition.
Re-audit metadata/group separation only; no search for more attractive splits.

## Exact matrix and common scientific settings

Order: `fold0_res224`, `fold0_res320`, `fold1_res224`, `fold1_res320`,
`fold2_res224`, `fold2_res320`. Every fit starts fresh official ImageNet V1
ConvNeXt-Tiny weights, SHA256
`983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`.
Never resume a v11/v9/demo checkpoint. All five backbone factories and both
historical loss implementations stay unchanged.

The only scientific contrast is224 versus320 pixels. Common seed2026, eight
independent logits, per-eye ratio-weighted BCE (fit-patient negative/positive
ratio only), FP32 AdamW lr1e-4→1e-6 cosine over25 epochs, weight decay .01,
no augmentation. Common microbatch8, accumulation2, effective16 images, no dropping
last samples. Scale each microbatch mean by its sample count divided by the actual
optimizer-window count, including the partial final window; one optimizer step per
window and one scheduler step per epoch. All training-mode stochastic layers and
normalization behavior are identical across arms.

V11 reference fits cannot replace the new224 primary arm: they used microbatch16
without accumulation, while both new arms use8×2. Matching effective batch size
alone does not guarantee the same stochastic masks, kernels, rounding or optimizer
trajectory. Their checkpoints/results remain cost/historical evidence. Rerun both
arms under the same new implementation. Do not use already-seen v11 outer errors
to select further changes; earlier adaptive choices still limit independence.

Checkpoint selection: maximum **inner patient macro AUROC**, earliest exact tie.
Early stop after six epochs without >1e-4 meaningful improvement, max25 epochs.
Save best and resumable last (optimizer/scheduler/loader/Python/NumPy/Torch/CUDA
RNG) after every full epoch and complete optimizer window. Strict hash/provenance
reload, including resolution, before resume. Never carry partial gradient windows
across a checkpoint. Outer inference is deferred until completion and strict best
reload; paused runs never assess outer patients.

Patient probabilities remain mean left/right sigmoid probabilities; raw p≥.5.
No threshold search, shared-temperature transfer, calibration, risk-policy fit,
old-validation use, final evaluation or full-data refit. REVIEW REQUIRED unchanged.

## Comparison and limitations

Primary contrast is mean paired outer macro AUROC (320−224); report per-fold and
pooled descriptive outcomes. Use fixed2000 paired containment-component bootstrap
replicates, seed2026, within fold with identical sampled groups for both arms.
Report defined/undefined replicate counts; suppress intervals if <90% defined.
Intervals condition on fitted models and omit training/selection variability.

Also report macro AP, macro/micro F1, all-eight precision/recall/F1/AUROC/AP,
positive/negative support and TP/FP/FN, especially sparse H. No partial-matrix
winner. If the AUROC interval includes zero, improvement is not established; if
F1 or rare-class outcomes worsen, reject blanket performance-improvement claims.
Do not alter thresholds, quality features, losses or reporting after outer results.
A numerical lead is at most a candidate for future study, not deployment or
independent confirmation. A new compatible independent cohort remains necessary.

## Cost, quota and stopping

Previous six-fit all25-epoch estimate2.783 worker-hours; area scaling gives
2.783×(1+(320/224)^2)/2≈4.23 hours, or5.50 with30% planning allowance. This is an
unverified throughput estimate, not charged quota. No new GPU benchmark is run.

Proposed strict **21,600 free GPU-quota seconds, max six sessions**, including
failures/resumes/overhead. First submission additionally requires the whole21600
seconds free and zero existing GPU reservation, so a known partial-quota start is
blocked. Recheck actual availability after any advertised refresh. Reserve2,700 seconds per224 fit (2,400 worker +300 guard)
and4,500 per320 fit (4,200 +300); six reservations exhaust the cap. No seventh
session or paid fallback. Sequential only. Before each push inspect owned jobs,
receipts, version status and actual free/used/reserved GPU quota. Never resubmit
ambiguous saves or completed fits. Record counter deltas separately from worker
time; SDK does not expose a guaranteed per-shape multiplier. Use the observed
prior TeslaT4 counters as historical evidence only. If actual charging/available
quota cannot cover the next reservation, stop and preserve an incomplete queue.
Runtime/deadline differences are operational limits, not different epoch budgets;
a paused arm invalidates a complete comparison until compatibly resumed within
both caps. Current quota is measured separately in readiness; do not assume refresh.

CPU preparation is separately bounded at1800 seconds, GPU disabled and internet
disabled on Kaggle. Training commands require the new exact approval record even
when CPU preparation has succeeded. No automatic monitoring after task completion.
