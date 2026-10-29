# Phase 6 protocol, fixed before calibration-label access

This prospective execution contract supplements the frozen v4 development protocol
and follows the completed Phase 5 validation-only backbone selection. PR #12 was
already merged at `5b059f2`, with passing final CI. The configuration is
`configs/reliability-v6.toml`; its commit/hash precedes opening calibration rows.
No historical split or duplicate decision changes. Final evaluation remains sealed.

## Predictors and stochastic capability

Use the selected ConvNeXt-Tiny epoch-two checkpoint, SHA256
`6871e49772543f3463f9d30dd6c758257022463bf562e21342fd154d7aa0eaf1`,
with unchanged EXIF/RGB 224 letterbox and ImageNet normalization. Primary raw
prediction is the mean of both eye sigmoid probabilities, **not mean eye logits**.
All eight independent labels with p >= 0.5 are returned. Patient-derived targets
remain unchanged, including differing targets inside containment components.

The actual trained factory and strict checkpoint inspection establish a p=0.2
Dropout at `head.0` followed by the eight-logit Linear head. Training enabled this
existing layer. Enable only this layer for 20 MC passes; do not add layers or enable
ConvNeXt stochastic depth. All LayerNorm, BatchNorm (if present) and other modules
remain in evaluation mode. Cache deterministic backbone features once, then apply
the trained stochastic head: this is exactly the declared head-only predictor,
not an approximation to full-backbone MC Dropout. Verify direct/feature-path
agreement and unchanged model state. Report unsupported capability rather than
inventing stochastic uncertainty if the expected trained layer is absent.

Per-pass seeds are 62026+s on validation and 72026+s on calibration, s=0..19.
Each pass processes the complete role's image feature matrix in frozen row order;
aggregate both eyes **within that pass** before variance. Save per-pass patient
probabilities, seeds and population variance (ddof=0). Head-only variability does
not represent all backbone/model uncertainty or establish clinical validity.
The MC mean is an auxiliary diagnostic, never silently the gating/correctness
predictor. Correctness always uses the deterministic patient probability, raw or
calibrated as explicitly named in each comparison.

## Confidence, technical quality and scores

Patient confidence c is min_k max(p_k,1-p_k). Uncertainty u is max_k Var_s(p_sk).
For calibrated comparisons first apply the same fitted temperature to each
patient-level stochastic pass, then calculate u; no second temperature is fitted.
The three fixed scores are c, c*(1-4u), and c*(1-4u)*q, clipped to their mathematical
[0,1] range for roundoff. No component weight or definition is selected using
outcomes, and adding uncertainty/quality may worsen results.

Quality is a technical heuristic on the canonical uint8 letterboxed image.
Trim complete outer rows/columns matching the exact ImageNet fill RGB (124,116,104),
then calculate the existing quality_proxies on the remaining rectangle: foreground
fraction (gray>8/255), foreground gray standard deviation, and horizontal plus
vertical squared-gradient energy. All-fill/empty content gives zero features.
A border coincidentally matching fill or sharp non-retinal artifacts can affect
these proxies; none is a clinical gradability or lesion-localization assessment.

Fit only the three empirical feature distributions on all frozen **training**
images, without labels: midrank ECDF F(x)=(count(<x)+0.5*count(==x))/n. A constant
feature gives a tie score of 0.5. Average three ECDF scores for each eye and use
the minimum of both eyes as q. No calibration/validation outcome adjusts these
normalizers. Keep feature arrays and normalizer source hashes privately.

## One shared calibration temperature

Follow v4/v3: for deterministic patient p, let l=logit(clip(p,1e-6,1-1e-6)); output
p_T=sigmoid(l/T). Thus temperature acts **after mean eye probabilities**. Fit one
T on calibration patients only, minimizing the mean eight-label binary NLL with
patient weight 1/(patients in that containment component), normalized by total
weight. Mixed-label components retain each patient's vector; never vote on labels.
This gives each component equal total mass. Require >=50 components and both
outcomes per class. No class-specific fit or calibration-policy subdivision.

Use bounded scalar optimization over log(T), T in [0.05,20], xatol 1e-8, maximum
500 iterations, with identity T=1 as a numerical reference. Record objective,
convergence and boundary hits; a failed/nonfinite/boundary solution blocks candidate
freeze pending review. Bounds are numeric safeguards chosen before label access.
Fit no second calibrator to MC means. Positive T preserves the p=0.5 decision
boundary; verify all deterministic binary decisions remain identical.

Calibration NLL/Brier and ten equal-width probability-bin per-class/macro ECE are
**in-sample fit diagnostics**. Also report component-equal and patient-equal NLL,
support and classwise Brier/ECE. Validation comparisons after this fit remain
conditional development evidence because validation already selected the model.
Neither set establishes held-out improvement or clinical calibration guarantees.
Sparse H calibration support remains a limitation even with pooling.

## Reliability comparisons and threshold rules

Produce all three fixed strategies for raw versus calibrated deterministic
probabilities. Use validation for the development coverage-risk comparison; save
calibration fit diagnostics separately. Calibration labels never fit a risk policy.
There is no supported clinical risk budget or independent policy partition in v4:
deployment remains **REVIEW REQUIRED for every patient**. Do not choose an automatic
acceptance cutoff or a winning reliability strategy from these same-set curves.

Exploratory curves use descending score thresholds and include complete tied-score
blocks. Report all unique attainable thresholds, plus prespecified target coverage
0,.1,.2,.3,.5,.75,1 (include all ties, report actual coverage). Zero coverage has
undefined risk. Primary descriptive risk is mean per-patient eight-label Hamming
error; also report any-label-error risk, all-eight class support, false-negative
counts/rates and F1 on accepted patients. Negative-label imbalance can make Hamming
risk look favorable. AURC is right-step sum of coverage increments times prefix
risk, including the first prefix from coverage zero; it is not a clinical guarantee.

Use 200 fixed-seed resamples of containment components for descriptive percentile
AURC intervals and paired AURC differences versus confidence-only within each raw/
calibrated predictor. Preserve all patients per sampled component. These intervals
are conditional, not adjusted for backbone selection and not sparse-class coverage
or clinical guarantees. Do not optimize thresholds, seeds, pass counts or quality
weights after observing results. Freeze all three candidate ablations, not a
post-hoc best strategy. Any later held-out final evaluation requires a frozen
candidate contract and explicit access authorization; it is not run in this task.

## Grad-CAM and compute/recovery

Select four validation patients by SHA256('62026|'+patient_id), independent of
labels/results, keeping both eyes. With all modules in eval, explain the class
having highest deterministic patient probability (ties class order), and report
**all** predicted-positive labels separately. If none are positive, identify the
target as below threshold. Differentiate the mean of both eye sigmoid probabilities
for that target through `backbone.features.7` (last spatial stage). Spatially average
activation gradients, channel-weight activations, ReLU, bilinear resize and normalize
each map to [0,1]; all-zero maps remain zero and are reported, never forced nonzero.
Save private maps, overlays, target/class probabilities, geometry and source hashes.
Heatmaps are explanations, not verified lesion localization or clinical adjudication.

One CPU input/quality job <=1800 seconds, one GPU inference/Grad-CAM job <=3600 seconds,
and one CPU calibration/analysis job <=1800 seconds, using free quota only. No
optimizer/backbone updates. Reuse completed train/validation cache and selected
checkpoint. CPU preparation opens only frozen calibration image rows after this
protocol commit; original data remains immutable. GPU inputs contain only cached
validation/calibration images and selected checkpoint, with no original dataset or
final manifest attached. Persist attempted submissions before API calls, job/version
IDs, nonces, source revision, package freeze and per-artifact SHA256. Never blindly
resubmit ambiguous jobs. Verify downloaded outputs and make private stage artifacts
recoverable from pinned private Kaggle versions. Commit permitted aggregates only.
