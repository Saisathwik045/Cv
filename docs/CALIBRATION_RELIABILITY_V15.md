# V15 post-freeze calibration and reliability research software

This prospective software protocol follows V14 main
`badbf8af8ef3c0ac57a942f70e594e9b8634a419`. Authorization covers source development,
CPU synthetic tests, documentation and one PR only. **No real-data analysis,
checkpoint download, model inference, Kaggle job or private-data access is
performed or authorized by this implementation.** Synthetic outputs must retain
`synthetic_fixture_not_real_v15_results`; they are never V15 research results.
Historical protocols, V14 results, frozen temperatures/thresholds and the demo
remain unchanged. All actual predictions remain **REVIEW REQUIRED**.

## Fixed candidates and trust boundary

Analyze both existing ConvNeXt-Tiny candidates together, in this order:

1. `v14_candidate_res320_seed2026`, output
   `saisathwikregula/trustretina-v14-fit-320-7bbe5ade/1`, best.pt SHA256
   `8a3341f2b416355d87a07196e5bc7825a81aa94b30d32e90cf0e65e6134c9bbd`.
2. `v14_reference_res224_seed2026`, output
   `saisathwikregula/trustretina-v14-fit-224-779a38cd/1`, best.pt SHA256
   `32a316e6c1a898a85249c4ef26df7440bb95d91c6667aecb508b722e17a43695`.

The public V14 artifact registry SHA256 is
`da8fc216144e5f233c4e58363632c53d7a43d1d8c6e07300ca366d965f7aafce`;
frozen-plan SHA256 is
`43351a86118e6465a0e24939faddecd9b09d8c8af49d1122c8f0f47f197749e9`.
These pins authenticate the recorded 34 output hashes, selected-checkpoint
identities, strict reload/state invariance/post-freeze inference evidence and
frozen v4 parent role manifests. No remote client, recursive artifact discovery,
model loader, `torch.load`, image decoding or inference is used by V15.
Checkpoints are opaque streamed SHA256 inputs only, under a future separately
approved analysis. No checkpoint bytes are accessed during this development.

`trustretina_v15.v15_inputs.load_bundle` verifies explicit, root-contained paths with
no symlinks or traversal. Every listed artifact is hashed before NPZ/JSON parsing;
NPZ uses `allow_pickle=False`. Both candidates and every role pass validation
before the first fit. Manifest SHA256, this protocol's SHA256, data kind, exact
run IDs and scope must match an explicit authorization mapping. Real use also
requires a nonempty separate user approval reference. These are fail-closed
execution guards, not a cryptographic approval service: the operator must obtain
actual user permission, never manufacture an approval record. This task creates
only synthetic-fixture approval mappings. The command-line tool exposes no
real-data fitting command.

## Roles and alignment

Class order is exactly **N,D,G,C,A,H,M,O**. Inputs are the unchanged TRAIN,
validation and calibration roles: 2,275 / 525 / 350 patients for real data.
Historical final and external roles are rejected. TRAIN rows are label-free,
used only for resolution-specific quality normalizers. Validation/calibration
rows retain their original eight binary patient-derived labels; unknown labels,
NaN, nonbinary and incomplete vectors are rejected, never made negative.

A future real-data manifest must supply original SHA256-pinned v4 role JSON
objects (`role`, `rows`) and derived eye-row lists. Original manifest labels,
patient/component/eye/image identities must match derived rows; TRAIN label
fields are dropped in its derived list. Original v4 role objects are already
role-specific; no whole-cohort or historical-final manifest is opened.
Both eyes appear exactly once per patient, with matching labels/components;
patients and containment components never cross roles. Mixed-label components
retain each patient's vector without voting. Both candidates must have identical
ordered patient/component/target panels. Duplicate image paths are rejected.

V14 prediction NPZ files and quality features have separate row orders. Predictions
are validated against rows sorted by patient then eye; quality is checked against
its exact original row order. Deterministic patient probabilities must equal the
mean of the two saved eye sigmoid probabilities. Saved patient labels, if present,
must equal the authoritative role labels. MC requires exactly 20 finite probability
arrays `(20,n,8)`, matching patient IDs and exact validation 62026..62045 or
calibration 72026..72045 seeds. Head-only scope and within-pass eye aggregation
are inherited from the hash-bound V14 producer; patient MC arrays cannot
independently re-prove their unavailable per-eye MC ancestry.

## Shared calibration and reporting

Fit exactly one new shared temperature per checkpoint, using only calibration
patients, never the old demo temperature. Reuse the unchanged
`reliability.development.fit_shared_temperature` algorithm: component-equal
mean eight-label patient NLL, weight `1/component_patient_count`, normalized by
weight sum; bounded log-T scalar minimization T in [0.05,20], xatol 1e-8,
maxiter 500, feasible identity reference. Require >=50 components and both
positive/negative outcomes per class. Optimizer failure, nonfinite loss or a
boundary solution blocks completion. These floors are numerical safeguards,
not adequate rare-class power. Record optimizer status, evaluations, identity
and fitted objectives and the in-sample interpretation; do not force a gain.

Apply `sigmoid(logit(clip(mean_eye_sigmoid,1e-6,1-1e-6))/T)` after eye aggregation.
Apply the same T to each saved patient MC pass before calibrated uncertainty.
No mean-eye-logit substitution, class-specific temperature, threshold search,
policy fit or model/strategy selection. Keep p>=.5. Assert unchanged binary
decisions; a numerical threshold crossing blocks analysis. Positive scalar
calibration cannot fix the observed FP/FN/F1 limitations.

Report raw and newly calibrated patient-weighted NLL, mean binary Brier and
macro 10-bin equal-width ECE, plus component-equal NLL/Brier. ECE is the mean
of eight class ECEs: each class uses sum(bin_count/n * abs(mean_probability -
positive_fraction)). Bins are [k/10,(k+1)/10), with 1 included in the last bin;
empty bins retain zero count and null values. Per-class reports include NLL,
Brier, ECE, bin counts/means/fractions and sparse/single-outcome flags. Log losses
clip to [1e-12,1-1e-12] for finite numerical reporting. Reliability diagrams show
all eight classes, both modes and explicit data-kind/role interpretations.

For both roles, disclose all-eight TP/FP/FN/TN, precision, recall, F1, FPR/FNR,
positive/negative patients and supplying components, macro/micro F1, subset
exact-match accuracy, labelwise binary accuracy and Hamming loss. Zero-division
precision/recall/F1 use zero with defined flags; undefined FPR/FNR are null.
Report signed 320-minus-224 differences and point-regression flags for every
class. Retain H/O false-positive and false-negative tradeoffs, D/A/M misses,
and V14 C/A/G regressions; no aggregate gain implies uniformly better behavior.

Reuse confidence c=min_k max(p_k,1-p_k), u=max_k population MC variance, and
fixed scores c, c*clip(1-4u,0,1), c*clip(1-4u,0,1)*q. Include MC means,
population per-class variances and binary predictive entropy. No backbone
uncertainty or clinical gradability claim follows from these signals.

Fit separate 224/320 quality ECDFs on TRAIN features only: mean three midrank
ECDFs per eye; minimum eye score per patient. Features retain V14 foreground
fraction, contrast and gradient energy. Persist sorted TRAIN distributions;
no demo normalizer is reused. Calibration/validation outcomes never tune quality.

Risk-coverage curves are retrospective descriptive prefix subsets, not actual
ACCEPT decisions. Include entire tied-score blocks; report Hamming and any-label
error risk, right-step AURC and fixed target coverage 0,.1,.2,.3,.5,.75,1 with
actual attainable coverage and per-class supports/FN/FNR/sensitivity/F1. Zero
coverage has undefined risk. Hamming risk can hide errors through negative-label
imbalance. For Hamming AURC use 200 paired whole-component bootstrap replicates,
seed 2026, with identical draws for both candidates; keep repeated component
multiplicity. Percentile 95% intervals are conditional descriptive uncertainty,
not independent confirmation, selection-adjusted bounds or a clinical guarantee.
All three strategies are retained regardless of worsening results.

## Outputs, execution boundary and next work

Calibration diagnostics are **in sample**. Validation diagnostics are conditional
on repeated checkpoint/model selection, never fresh final evidence. There is no
independent evaluation endpoint or new scientific performance result in this PR.
Sparse calibration H (10 positives), validation H (16), dependent/mixed components,
patient-level label inheritance and unresolved H/O/external semantics remain visible.
BRSET access/DUA/credentialing/label compatibility and independent all-eight
confirmation remain separate blockers; no external access request is submitted.

`analysis.json` contains temperatures, provenance, source/package hashes,
normalizers, signals, curves and both-model comparisons. It and all generated
figures are private artifacts, never automatically committed or uploaded.
`output-hashes.json` records every output digest. Use a fresh output directory;
refuse overwrite. A failed call cannot silently publish a complete report.

Working commands (no accounts, private artifacts or real models):

```sh
.venv/bin/python scripts/calibration_reliability_v15.py readiness
.venv/bin/python scripts/calibration_reliability_v15.py synthetic-check
# Optional clearly marked synthetic diagrams in fresh temporary output:
.venv/bin/python scripts/calibration_reliability_v15.py synthetic-check --output /tmp/trustretina-v15-fixture-report
.venv/bin/python -m unittest discover -s tests -p test_v15.py -v
```

A future real-data request needs explicit separate authorization, a reviewed
hash-bound local manifest, permitted artifact recovery and original role-manifest
verification. The manifest's top-level keys are `data_kind`, `protocol_sha256`,
`class_order`, `run_ids`, `registry_file`, `frozen_plan_file`, `roles`, and `runs`.
Each role has `file`, `sha256`, `parent_manifest_sha256`, `original_file`; each
run has `directory`, `kernel_version`. Paths are relative to the manifest root.
Authorization keys are `approved`, `scope=v15_cpu_shared_temperature_analysis`,
`manifest_sha256`, `protocol_sha256`, `data_kind=real_v14`, exact `run_ids`, and
`user_approval_reference`. No real approval template is enabled or written here.
Direct library analysis is `trustretina_v15.v15.analyze(..., fixture=False)`; do not call
it on real inputs under this software-only authorization. No network action is
part of that library. No deployment candidate is promoted and no demo is replaced.
