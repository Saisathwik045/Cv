# Next-stage candidate protocol — planning only

This prepares a reproducible 320-pixel ConvNeXt-Tiny ratio-BCE candidate and a
matched 224 reference. **No new model, temperature, thresholds, splits or demo
bundle is fitted or changed. No GPU job is authorized by this document.** The
original five backbone implementations, both losses and REVIEW REQUIRED behavior
remain unchanged. [Exact proposal](../results/candidate-v14/proposed-plan.json),
[dataset feasibility](INDEPENDENT_DATASETS_V14.md), [recovery verification](../results/candidate-v14/recovery-verification.json)
and [cost derivation](../results/candidate-v14/compute-estimate.json) are review inputs.

## Evidence and recommendation

Merged PR #25/main `fe808b24c438503ba89462a3ed206fb7f3ae484d` contains all six
completed v13 fits. Prior head `26e74ba` passed seven CI jobs in
[42776568639](https://github.com/Saisathwik045/Cv/actions/runs/42776568639).
The [unchanged v13 discussion](../results/resolution-v13/DISCUSSION.md) is the
source of all results below, not screenshots or a rerun.

320 mean outer macro AUROC 0.85108 versus 224 0.83273; paired difference +0.01835,
conditional 95% interval [+0.01036,+0.02655]. Macro AP 0.57279→0.59412,
macro F1 0.53758→0.57156 and micro F1 0.53534→0.57509. Retain 320 as the leading
**development** candidate, not an independently confirmed improvement.

| Prespecified weakness,224→320 | Retained evidence |
|---|---|
| H | Recall 0.1791→0.2537; precision 0.3243→0.1828; F1 0.2308→0.2125; TP 12→17, FP 25→76, FN 55→50. Recall fell in two folds; paired H recall/F1 intervals include zero. |
| O | Recall 0.4450→0.6588; precision 0.3920→0.3679; FP 439→720, FN 353→217. One fold's recall fell. |
| D/A/M | FN 269→316,62→66,11→19 respectively at p≥.5. |
| G | Pooled AP 0.5203→0.4735 and F1 0.5442→0.5361. |

Sparse H support, per-eye inheritance of patient labels, co-occurrence, image
quality and source semantics remain limitations/hypotheses, not established sole
causes of the errors. More resolution does not resolve these automatically.

**Recommended next executable stage:** resolve external access/endpoint feasibility
and prepare the missing CPU-only role inputs/worker freeze before seeking GPU
execution approval. Start with permitted BRSET metadata for a partial-label study;
no full-eight independent dataset is verified. New 320 validation/calibration
caches must come from original native images. Reuse all completed TRAIN
preparation, v4 partitions, duplicate containment, source hashes and training
artifacts. Do not run another smoke test, change loss, search seeds or train five
backbones. The two fits create standalone models on the whole permitted TRAIN role and a
matched control for later transport evaluation. They do not repeat the completed
nested study. More training patients is a reason to test, not an established
rare-class improvement. This is a freeze protocol, not another architecture search.

## Exact proposed fits and immutable data roles

| Order / run ID | Resolution | Fit / checkpoint selection | Max epochs | Worker timeout / reservation |
|---|---:|---|---:|---:|
|1 `v14_candidate_res320_seed2026` |320|Frozen v4 TRAIN 2,275 / validation 525|25|7,500s /7,800s|
|2 `v14_reference_res224_seed2026` |224|Same TRAIN / validation patients|25|3,900s /4,200s|

Use the exact v4 release
`c094f372aa8e9e2b15d1483c8f0f9cab58f3e3beabc11e938fd281f205c7af57`.
The proposal records exact train/validation/calibration manifest hashes; verify
original patient/component IDs and all-eight positive/negative support against
them before constructing a request. Both eyes and every containment-connected
patient remain together across all roles. No new allocation or favorable split
search. The older three v13 folds remain historical nested assessment artifacts.
Their checkpoints fit only 1,137 patients each and cannot substitute for a matched
2,275-patient reference. Do not pick the best outer fold or average fold metrics
as if they were a new independent evaluation.

| Role | Patients | Positive patients N/D/G/C/A/H/M/O |
|---|---:|---|
|Fit: TRAIN|2,275|738/733/140/138/107/67/113/636|
|Checkpoint selection: validation|525|175/169/32/32/25/16/26/147|
|New model-specific shared temperature only: calibration|350|116/113/21/21/16/10/17/98|
|Historical final|350|Excluded; labels/images/predictions not used in this stage|
|Independent final|Not yet identified|Must be registered/checked separately; no invented support|

These counts come from committed training/Phase6 aggregate support, not reopening
historical final or calibration patient records. Negatives are included in the
machine-readable proposal. Calibration has only 10 H-positive patients; one
shared scalar is the design, not class-specific calibration or policy fitting.
The calibration role was used for the original model and is not a fresh cohort.
Validation already influenced earlier choices. Neither role can support new
independent confirmation, even if new results look better.

Every fit starts fresh official ImageNet V1, SHA256
`983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`,
never a demo, v9, v11 or v13 checkpoint. Identical settings except resolution:

- Existing eight independent logits and trained p=0.2 dropout head; class order N,D,G,C,A,H,M,O.
- Ratio BCE positive weight=negative/positive **TRAIN-patient** counts only, inherited by both eyes; no weight estimates from validation/calibration/external cohorts.
- Native RGB/EXIF mean-color letterbox, bilinear resizing, ImageNet channel mean/std; no new crop, augmentation or cache upscaling.
- Seed 2026; TRAIN loader shuffle uses the recorded seeded generator; validation/calibration order is sorted patient ID then left/right. Save batching/order metadata. FP32 AdamW, lr1e-4→1e-6 cosine over 25 epochs, weight decay 0.01; microbatch 8, accumulation 2, effective 16 with exact sample weighting for partial windows.
- Checkpoint selection maximizes validation **patient macro AUROC**, earliest exact tie. Early stopping uses the same meaningful-improvement reference/min_delta 1e-4/patience 6 rule as v13; tiny strict improvements may update best without resetting patience. No seed or checkpoint choice from calibration/external results.
- Save best and last every epoch with optimizer, scheduler, Python/NumPy/Torch/CUDA and loader RNG, source/dependency/input/role hashes. Strict reload before selected-checkpoint predictions. Do not refit on TRAIN+validation after selection.

The existing v13 runner deliberately rejects this new schema/data-role budget.
Do not edit it or bypass approval. A new reviewed worker/registry must enforce the
new roles, emit source/config/input hashes and fit-only weights, and forbid the
historical final. Its code and actual missing input hashes are **not yet frozen**;
null fields in the proposal are real blockers, not placeholder evidence of inputs.
Freeze the actual dependencies/Docker revision used by v13 unless separately
reviewed; CUDA/Torch/device-count drift stops compatible resume. The partial-label
mask-aware external evaluator also needs implementation/tests before use.
TRAIN224/320 hashes are already known and recorded. Recover existing 224 role
artifacts; only missing320 validation/calibration tensors require new CPU
construction from hash-verified native originals. Existing224 train/validation
storage is `saisathwikregula/trustretina-full-v5-inputs-4f23af6d/1`;
existing224 calibration inputs are
`saisathwikregula/trustretina-phase6-inputs-0419699f/1` (recover only needed
role inputs, not old fitted temperatures or historical final data). Their existing
receipts/indices are the recovery authority; current availability of these older
role caches was not rechecked in this task. Propose at most one GPU-disabled CPU
preparation job, capped at3,600s, for the missing320 validation/calibration tensors
only. No such job is submitted here. Current local free disk is about276MiB,
insufficient for new role caches/checkpoints; verify adequate free private scratch
or a reviewed streaming/pinned-output recovery route first. Do not delete verified
artifacts or assume paid storage. Stop on a mismatch; never
substitute metadata for actual image tensors or recreate partitions.

## Freeze order and calibration separation

1. Persist reviewed source/configuration/input/role hashes and a new exact
   execution approval before any submission. TRAIN and checkpoint-selection
   validation labels are the only labels available to optimization. Calibration
   inputs, if packaged for later inference, carry image IDs/hashes without labels
   in the training worker.
2. Complete each fit and strict best reload; persist its selected checkpoint hash,
   best epoch, predictor/preprocessing contract and raw validation predictions.
   Model selection is now closed. Do not use calibration/external performance to
   pick among runs, epochs, resolutions, strategies or image cases.320 remains
   the nominated candidate;224 is a disclosure/control reference.
3. Only after that checkpoint freeze, compute deterministic calibration predictions
   and20 trained-head MC passes with both eyes aggregated within each pass. MC is
   head-only with cached deterministic features, normalization/stochastic depth
   in eval; no inserted untrained dropout. Save validation seeds62026..62045,
   calibration72026..72045, future external92026..92045 and the ordered batch/ID
   contract. No outcome-driven pass count or strategy selection.
4. A separately gated **CPU** analysis may fit exactly one shared temperature per
   new checkpoint on the350 calibration patients. Reuse the existing
   `fit_shared_temperature` algorithm: component-equal eight-label patient NLL,
   log-T bounded scalar optimization T∈[0.05,20], xatol1e-8/maxiter500; retain
   boundary/optimizer/support failures as blockers. Apply
   `sigmoid(logit(clip(mean_eye_sigmoid,1e-6,1-1e-6))/T)`, not mean eye logits.
   Report fit diagnostics as in-sample. Do not reuse T=0.722602 from the old model
   or alter its bundle. No class-specific T, disease-threshold fit or acceptance
   policy. At p=.5, positive shared T cannot repair FP/FN/F1; ranking is also
   monotonic apart from clipping ties. Calibration is not a substitute for
   improving discrimination/rare-class detection.
5. Retain the existing confidence, head-only uncertainty and technical quality
   definitions. Fit quality ECDF normalizers on TRAIN only, separately at224/320;
   never transfer the old224 normalizer to320 or fit them on an external cohort.
   Quality is a technical proxy and fusion previously had negative findings;
   no gating strategy is selected. Save the scalar T, normalizer, logits,
   stochastic arrays, dependencies and every artifact SHA256 privately.
6. Freeze a new research contract containing checkpoint/T/normalizer/preprocessing/
   class-order/patient-aggregation/seed/threshold/source hashes and default
   **REVIEW REQUIRED**. Preserve the original working demo entirely. Independently
   approve a dataset-specific evaluation request only after its eligibility,
   labeling, overlap, license and support gates pass. Its predictions/results
   must never update this contract. A new experiment after exposure requires a
   new version and fresh confirmation strategy.

## Prospective evaluation and claims

Primary endpoint: **raw deterministic mean-eye patient macro AUROC difference,
320−224**, on a prospectively registered independent cohort with all eight
compatible labels and positive/negative support. AUROC uses one-vs-rest labels;
macro AP means the unweighted mean of eight average precisions, not trapezoidal
PR area. Both models predict exactly the same eligible patients. Ties in AUROC/AP
use the library's standard tied-score handling. No label argmax or softmax.

At the fixed decision p≥.5, report all-eight precision=TP/(TP+FP),
recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN), TP/FP/FN/TN, positive/negative patients,
predicted positives and positive/negative containment components. Macro F1 is
mean class F1; micro F1 uses total TP/FP/FN on the equal-shaped(n,8) binary arrays.
Use existing zero-division0 display conventions while flagging undefined
precision/recall denominators; undefined AUROC/AP remain NA. A full-eight ranking
endpoint requires all eight classes defined, never an unnoticed seven-class mean.
Report labelwise binary accuracy, Hamming loss and all-eight subset exact-match
accuracy under those names, never diagnostic accuracy.

Primary point estimates weight patients equally. Secondary estimates give each
containment component equal total weight using inverse component size. Use 2,000
fixed seed 2026 paired **whole-component** bootstrap draws, with all patients of
each sampled component and identical draws for both models, separately per
registered dataset. Recompute patient- and component-weighted metrics each draw;
percentile95% intervals require at least 90% defined replicates and include the
undefined counts. Do not silently bootstrap eyes/patients independently or pool
unrelated datasets with incompatible labels. Intervals condition on fitted models
and miss training/selection uncertainty. For cluster multiplicity m_g and original
eligible component size n_g, patient-weighted bootstrap weights are m_g and
component-weighted weights are m_g/n_g. Do not renormalize duplicated IDs in a way
that cancels resample multiplicities. Partial-label secondary weights use the
known-label eligible count within each component for that class. Per-class intervals are descriptive,
not multiplicity-adjusted class-superiority guarantees.

A primary difference interval wholly above zero supports a ranking difference
on that eligible cohort only. Report macro AP/F1/micro F1 alongside it. For H/O
always report FP **and** FPR=FP/(FP+TN), FN/FNR, precision, recall, F1 and supports;
for D/A/M explicitly report missed positives/FNR and recall changes. Any increase
in sentinel FPR/FNR, decrease in recall or class F1 is a visible regression flag,
with its interval. Separate a point regression from interval-supported evidence;
there is no invented clinical noninferiority margin. An aggregate gain cannot
support a broad claim of uniformly better predictive performance while these
tradeoffs remain. Never select a model/threshold/strategy or reporting cases from
independent results. Include all eight rows, not only favorable classes.

Compare raw/newly calibrated NLL, Brier and 10-bin equal-width ECE/reliability
plots on the independent known-label set, with paired grouped differences and
in-sample calibration fit diagnostics clearly separated. No new risk policy,
ACCEPT cutoff, clinical recall promise or deployment guarantee.

Do not select high-quality images after examining errors. Any source gradability
restriction/unknown-label rule must be registered before inference, with all
excluded/missing counts reported; technical quality remains an experimental signal.

For partial datasets, use a registered native-label mask: unknown entries never
become negative. Report per-target scores/support and explicitly named partial
macro/micro summaries on known cells only. All-eight subset accuracy is NA unless
the entire patient target panel is known. No all-eight claim is permitted from
BRSET/PAPILA/RFMiD/mBRSET/EyePACS as currently verified. Probability losses against
a mismatched native proxy are transport descriptions, not validation of ODIR
calibration; never refit an external temperature to hide that mismatch. A prospective full-eight
cohort and taxonomy agreement remain blockers even if partial results are good.

## Proposed budget, recovery and execution boundary

[Measured-history extrapolation](../results/candidate-v14/compute-estimate.json):
median epoch 53.317s at224 and 104.490s at320; scale by 2.2 for the larger frozen
TRAIN/validation workload and use the maximum observed non-epoch overhead.
All 25 epochs estimate 8,947.42 worker seconds (2.49h). Applying the prior observed
charged/worker ratio 1.0312 gives about 2.56 quota hours; with 25% allowance, about
**3.20 hours**. This is an estimate, not measured full-role throughput or a billing
multiplier guarantee. Bounded post-selection/calibration inference must fit the
same timeouts; twenty head-only passes use cached features without repeated
backbone inference.

Propose a **strict 14,400 free Kaggle GPU-quota seconds (four hours), maximum three
submitted GPU sessions**, including failures, resumes, provisioning and inference.
Initial candidate/reference reservations 7,800+4,200=12,000s. At most one compatible
resume has worker ≤2,100s plus 300s overhead, total remaining reservation ≤2,400s.
No fourth session or paid fallback. No external-evaluation GPU job is included;
its dataset/size/access are unknown and need a separate bounded request.

Read-only observation currently shows 14,673.105s free, 0 reserved, paid disabled;
no quota was reserved. Recheck the full proposed cap immediately before starting
and again before each submission, plus actual accelerator shape/charging.
Record a new account-counter baseline/refresh window and conservatively count
all observed usage increases, including unattributed concurrent activity. Never
reset the baseline on resume or equate worker elapsed time to billed quota.
Stop if quota/reservations/counter rate cannot cover the next session.

Persist a new registry and receipt **before** each push. Inspect owned remote jobs
and exact versions first; reuse complete artifacts after checksum verification.
On interruption, restore only a checkpoint with matching source/config/input/
role/initialization/optimizer state; record a new attempt against the same fit and
remaining global/session budget. Save epoch checkpoints, RNG and source/nonce
receipts incrementally. Poll only at bounded intervals (at least 180s). An
ambiguous push, missing checkpoint, refresh-window change or attribution issue
stops execution for reconciliation; no blind replacement. If a fit cannot finish,
keep successes/failures, report the incomplete matched pair honestly and do not
declare a new paired winner. Preserve private durable pinned output versions and
public permitted aggregate hashes; uploaded patient data/checkpoints stay out of
Git. Save exact resume commands and promise no background monitoring after closeout.

The present authorization covers research/read-only verification/plan preparation
only. Missing actual input hashes, new worker implementation/source freeze,
licensed endpoint-ready external data, review and separate budget approval are
explicit gates. No training, calibration, threshold fit, new image preparation,
external inference, data upload, paid resource or demo replacement occurred here.
