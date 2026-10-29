# One frozen held-out research evaluation, Phase 7

This protocol/configuration is committed **before opening final role rows,
images or labels**. PR #13 is already merged at `c4d563a`; final CI for its head
`e13a4bf` passed. The unchanged v4 split, Phase 6 candidate contract, selected
checkpoint, fitted temperature and training-derived normalizer are hash-verified.
No refit, backbone training, seed search, new split or duplicate review is permitted.
The demo application is out of scope. All aggregate results, including unfavorable
ones, must be preserved. Final exposure is authorized only for this bounded study.

## Immutable predictor and scope

Contract SHA256: `a80dce8f27f12645539a0312a3ed96af0b5d10b5d7e1cd522af53b9347409c61`.
Checkpoint SHA256: `6871e49772543f3463f9d30dd6c758257022463bf562e21342fd154d7aa0eaf1`.
Root v4 manifest SHA256: `c094f372aa8e9e2b15d1483c8f0f9cab58f3e3beabc11e938fd281f205c7af57`.
Use all 350 final patients, both eyes and every frozen containment component.
Verify exact patient/eye/label/component alignment; preserve each patient's own
source vector inside mixed-label components. No voting, deletion or exclusions.

Use unchanged 224 EXIF/RGB letterbox and ImageNet normalization. Deterministic
patient probabilities p are mean(left/right sigmoid outputs). Calibrated output
is sigmoid(logit(clip(p,1e-6,1-1e-6))/T), with the saved T and saved parameter-file
hash. Do not average eye logits. Use all eight independent labels at p >= 0.5;
argmax is never the multilabel prediction. No class-specific calibration.

Reuse exactly the frozen validation MC sequence **62026..62045**, twenty trained
head.0 Dropout p=0.2 passes on the whole final image-feature matrix in frozen order.
This final-role choice between the two existing role sequences is made here,
without final-data access; no new seed is searched. Backbone stochastic depth and
normalization remain eval. Both eyes aggregate within each pass. The stochastic
mean is diagnostic only: correctness and confidence use the deterministic patient
predictor. Calibrated pass probabilities receive the same saved T after each eye
mean. No second fit. Verify model parameters/buffers unchanged.

Apply the saved training feature ECDFs without fitting any new distribution.
Use exactly Phase 6 canonical quality features, min-eye patient quality and the
three score definitions: c=min_k max(p_k,1-p_k), u=max_k Var_s(p_sk), then c,
c*(1-4u), c*(1-4u)*q, raw and calibrated. Retain all six comparisons; select no
strategy, threshold or reporting case from final results. No final image examples
or Grad-CAM case panels are selected; aggregate reports cover every class/strategy.

## Endpoints, definitions and weighting

**Primary endpoint:** patient-equal raw macro AUROC, the arithmetic mean of eight
classwise AUROCs. Each class must have both outcomes; otherwise macro AUROC is
undefined, never computed by silently dropping a class. AUROC uses the standard
weighted concordance/rank implementation, including half credit for score ties.
Raw/calibrated per-class AUROC and AUPRC (step-integral average precision),
precision, recall, F1, predicted-positive/positive/negative counts are secondary.
AP is undefined if either outcome is absent, matching the existing project rule.
Precision/recall/F1 undefined denominators use zero, with denominator flags/counts
visible; ranking undefined values are null. Macro F1 averages all eight class F1s.
Micro F1 is 2*sum(TP)/(2*sum(TP)+sum(FP)+sum(FN)) on the (n,8) arrays, not argmax
or flattened single-class F1. Report macro precision/recall and micro precision/recall.

Report subset exact-match accuracy (all eight decisions correct), labelwise binary
accuracy (mean fraction of correct binary labels per patient), Hamming loss (one
minus labelwise accuracy), and any-label-error rate (one minus subset accuracy).
Report patient/class NLL with probability clipping 1e-12 for logarithms, Brier
mean squared probability error, and fixed ten equal-width-bin class ECE (bin mass
weighted absolute mean-probability versus prevalence gap); macro ECE averages the
eight ECEs. Empty-bin means are null. Publish raw/calibrated reliability diagrams.

Primary estimates give every patient weight one. Also report component-equal
metrics with weight 1/(number of final patients in that frozen component). Each
component has total mass one; mixed-label patients retain their vectors. Weighted
classification uses weighted TP/FP/FN and weighted ROC/AP. Weighted NLL/Brier/ECE
normalize by total patient weight. Weighted support mass is distinct from integer
patient counts and positive/negative supplying-component counts. These weights
are evaluation estimands, not a claim of clinical independence or a refit.

## Coverage and risk: retrospective research only

Use the unchanged coverage grid 0,.1,.2,.3,.5,.75,1. **Grid cutoffs are the exact
saved Phase 6 validation cutoffs for each of the six strategies**, not quantiles
estimated on final predictions or labels. Report nominal validation-grid coverage
and actual final patient/component-weighted coverage, which can differ. At nominal
zero select no patients, with risk undefined. Include all patients tied at each
cutoff. The full-partition endpoint is separately reported even if the saved
validation minimum cutoff does not cover all final patients.

Also report each complete retrospective final coverage-risk curve over **all**
attainable score levels, without selecting a useful-looking level. This is an
evaluation of a predeclared ranking functional, not threshold selection for use.
Tied scores form indivisible blocks. Primary risk is mean per-patient eight-label
Hamming error; secondary risk is any-label error. Patient coverage is accepted
patients/all patients. Component-equal coverage is accepted weight/total weight;
weighted risk averages fixed full-partition weights in the retained subset.
Partial retention of a component does not redefine its original weights.

AURC is the right-step sum of coverage increments times prefix risk, including the
first observed prefix from coverage zero. Zero-coverage risk itself is undefined.
For each saved-grid subset report all-eight positive/negative/predicted-positive
support, false-negative count/rate, sensitivity, precision and F1 (plus weighted
versions). FNR/sensitivity are null when no positives occur. These subsets and
curves are **retrospective research analyses, not a validated acceptance policy**.
Deployment remains REVIEW REQUIRED for every patient; no ACCEPT threshold exists.

## Fixed uncertainty analysis

Use 200 resamples, seed 62026, drawing the same number of whole containment
components with replacement and keeping all patients in each draw. Use the same
resample indices for raw/calibrated and all six scores. Do not refit anything.
For the patient-equal estimand, patients in each sampled occurrence have unit
weight. For component-equal estimates, reuse each original row's 1/group-size
weight on every drawn occurrence: each drawn component retains mass one. Never
recompute group counts by original ID across repeated draws, which would cancel
bootstrap multiplicity.

Report percentile 95% intervals for primary/secondary classification and calibration
metrics, AURCs, and saved-grid coverage/Hamming/any-error risks. Report paired
calibrated-minus-raw NLL/Brier/ECE differences, including classwise diagnostics,
and paired AURC differences versus confidence-only and calibrated-minus-raw.
Use only defined resample statistics and state valid/total replicate counts;
require >=160/200 defined values for an interval, otherwise null. Macro ranking
requires all eight labels defined in that replicate. Per-class sparse support,
zero-positive retained subsets and undefined metrics remain explicit. Intervals
are conditional on this frozen candidate, not simultaneous/multiplicity-adjusted
clinical guarantees; 200 replicates provide limited tail precision. No p-value or
clinical risk certification is inferred. Sampling representativeness, inherited
patient labels, known component dependence and unknown identity overlap remain
limitations. This is one internal held-out cohort, not external clinical validation.

## Compute, recovery and defect handling

One free Kaggle CPU input job <=1800 seconds, one GPU inference job <=1800 seconds,
and one CPU evaluation/report job <=1800 seconds. Original images remain immutable.
CPU preparation reads only explicitly authorized frozen final image rows. GPU
inputs contain only the derived final cache and selected checkpoint, with no
original dataset or development image rows. Analysis receives frozen Phase 6
parameter/normalizer/validation-cutoff artifacts and final predictions; it has no
optimizer or fit path. Record package/image versions and all source/config hashes.

Persist write-ahead submission receipts, nonces and job/version IDs. Reconcile
remote jobs before submitting; never blindly resubmit or repeat a completed stage.
Record final-manifest exposure after this protocol commit in an append-only private
ledger. On implementation failure, preserve receipt, source, logs, partial outputs
and hashes; record the defect, correction and extent of final exposure **before**
any explicitly marked corrected attempt. Do not overwrite failed versions or use
results to modify statistical choices. GPU/CPU caps include any corrected attempts;
a substantive validity problem or exhausted budget stops execution with progress
saved. Retrieve/hash-verify private outputs and fresh-task recovery pointers.

Private predictions, IDs, images and full attainable-score thresholds remain out
of Git. Publish permitted aggregate tables/figures and a concise research report
generated from verified artifacts, covering all classes and fixed strategies.
The next phase packages this same frozen pipeline into the demo app; it does not
retune this final evaluation or add a clinical acceptance policy.
