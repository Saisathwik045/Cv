# Development protocol v3 — bounded verification and conditional execution pilot

This revision follows merged main `0d5986f`, which contains PR #8's independent
photo-review policy and PR #9's Phase 2 CPU modules. It preserves the pipeline and
all five backbones. Earlier weak-dHash connected splits and blanket-quarantine
sensitivity manifests remain unchanged and are not approved by this document.

## Bounded automated photographic review

The source policy was fixed at commit `2da502a` before the real CPU execution.
The job is private, CPU-only, attached to original dataset version 2, and limited
to 1,800 seconds and 1,536 detailed image comparisons. It performs no fitting.
All referenced raw hashes, decoded EXIF RGB hashes, independent dHashes and the
original authoritative workbook/label/reference digest are checked. Retinal
foreground intensity and gradient comparisons rank candidates; neither those
scores nor dHash alone establishes a duplicate.

The 302 earlier technical/bridge candidates and 64 rejected controls are included.
Up to 128 neighbours retrieved by retinal foreground content outside the dHash
candidate set add a bounded false-negative check. Remaining detailed slots alternate
foreground intensity and gradient similarity rankings, with deterministic ties.
The class labels, partition sizes and model results do not enter this ranking.

Detailed review uses 512-pixel retinal crops, contrast-enhanced SIFT descriptors,
mutual ratio matches and RANSAC alignment. Original v2 photometric/geometric copy
criteria remain fixed: >=12 inliers, >=.85 inlier fraction, >=.15 spatial coverage,
>=.70 overlap, aligned foreground SSIM >=.95 and normalized MAE <=.02. Increased
resolution and feature contrast address previously observed feature scarcity;
they are not changes chosen for class support or model performance. Four
content-hash-ordered informative real sources generate resize/JPEG90, mild-exposure
and black-padding controls; low-information images remain unresolved. Failed
controls disable new probable-copy links.

Raw/pixel identity confirms duplicates. Independent registered detail supports
probable copies. Verification failure alone never becomes an automated false
positive. Previously verified copy links remain links even if a later method
fails. Contradictory review decisions are explicit blockers. Prior supported
false positives retain their evidence; no clinical or patient-identity adjudication
is invented. Raw images and original patient-derived labels are never changed.

Confirmed/probable links form transitive patient groups, always keeping both eyes.
Other weak candidates remain private sensitivity records, rather than automatically
joining a giant component or removing most of the cohort. Specifically material
uncertainty includes foreground correlation >=.98, aligned foreground SSIM >=.90,
>=12 geometric inliers with >=.5 consistency, and previously identified possible
copies that remain unresolved. These are **review triggers**, not copy thresholds.
Their patient IDs/eyes and evidence are recorded privately. A crossing of such a
case between development roles blocks split freeze and pilot submission. A bounded
pass cannot establish absence of every missed duplicate or clinical identity.
Any exhausted/incomplete pass also blocks readiness; it does not silently approve
unprocessed edges.

## Four-role prospective protocol

Use the original group-aware rarest-label allocator and whole-group support repair
`trustretina-component-rarest-label-repair-v3`, seed **2026**, targets .65/.15/.10/.10
for train/validation/calibration/final evaluation. The implementation is shared
with historical allocation, but the input graph now contains supported links only.
No alternative seeds or model results are searched. Repeating the same assignment
only verifies determinism. Labels are N,D,G,C,A,H,M,O from original data.xlsx;
these are patient-derived labels, not verified individual-eye diagnoses.

Report every class's positive/negative patient counts, positive/negative connected
group support, mixed-label groups and component sizes. Minimum five positive and
five negative patients **and groups** per role is an operational development floor,
not evidence of adequate power. Conflicting annotations on supported copies require
review; labels are neither voted on nor corrected automatically. No default blanket
quarantine cohort replaces the original population. No split is frozen if material
uncertainty crosses roles or input/support/separation/determinism checks fail.

Checkpoints and backbones are selected using validation only: primary patient
macro F1 at fixed .5, ties earliest epoch then the original five-backbone order.
Mean left/right sigmoid probabilities before thresholding; eye-level outputs are
secondary. Disease thresholds remain .5 in this simpler design. Any future
validation threshold-search design needs a separate prespecified amendment.
Final evaluation images/labels cannot be used for checkpoint/backbone/threshold
selection, and the execution pilot cannot open them at all.

## Simpler calibration without removing architecture components

Reserve one calibration partition. After validation-only model selection, a later,
separately authorized fit may estimate **one shared scalar temperature**, applied
to logits of clipped mean eye probabilities (clip 1e-6), using group-weighted
multilabel NLL. Require >=50 independent groups and both outcomes for every label.
There are no per-class temperatures and no internal calibration-policy subdivision.
Rare classes do not gain class-conditional calibration guarantees from pooling.
The count floor is a numerical/development safeguard, not a power justification.
Per-class support and uncertainty must still be reported honestly.

Confidence, MC Dropout, technical quality, calibration and ACCEPT/REVIEW interfaces
all remain. This simpler development design does **not fit a risk/coverage cutoff**
on the same calibration examples. The reliability policy stays unfitted and returns
REVIEW REQUIRED; no clinical acceptable-risk budget is invented. Calibration-set
fit statistics are descriptive, not held-out calibration performance. Future policy
fitting or cross-fitting requires its own support/dependence and risk-confidence
protocol. Final evaluation stays untouched until the complete pipeline is frozen.
Use patient/duplicate-group resampling and exact sparse-class numerators/denominators;
report wide intervals or undefined metrics instead of unsupported rare-class claims.

## Conditional single GPU pilot

User authorization is for one free-quota EfficientNet-B0 execution pilot, at most
two epochs, using a fixed training-only subset. The declared cap is 128 training
patients (both eyes) and 64 validation patients; whole supported-copy groups stay
intact. Select subset groups by seed/content-derived order, never model outcomes.
Use 224-pixel shared preprocessing, batch 16 and AdamW 1e-4. Random initialization
is explicitly execution-only; the planned full ImageNet-initialized comparison is
unchanged. The pilot must not select models or fit calibration/reliability parameters.

Only a successful real-data readiness audit can create a new immutable four-role
release. A failed audit writes a private blocked-pilot record with no job ID or
submission attempt. It never publishes a `complete` marker or overwrites a historical
manifest. This is separate from Phase 2's full-experiment approval guard; simplifying
calibration does not silently turn on five-model training.

If readiness passes, the sole pilot must verify finite forward/backward losses,
save an epoch-boundary checkpoint, strictly reload model/optimizer/RNG/provenance,
and finish the remaining authorized epoch from that checkpoint. Persist its job ID
before polling; an ambiguous submission may only be reconciled/resumed, never
blindly repeated. Retrieve predictions/logs/checkpoints with source/nonce/checksum
verification. All pilot outputs are execution evidence, not final performance.
No final rows are packaged into a pilot request. No GPU job is permitted while the
readiness audit is blocked.

## Executable checks and private artifacts

```sh
trustretina check-development --config configs/development-v3.toml
python -m unittest discover -s tests -v
```

The existing verified Kaggle workflow retrieves a pinned CPU job's report and
output checksums. Then consume the private evidence with:

```sh
PYTHONPATH=src python scripts/prepare_development_v3.py \
  --inventory /private/inventory.json --inventory-sha256 TRUSTED_INVENTORY_SHA \
  --report /private/independent-report.json --report-sha256 TRUSTED_REPORT_SHA \
  --receipt /private/verification.json \
  --prior-report /private/v2/independent-report.json \
  --prior-report-sha256 TRUSTED_PRIOR_SHA \
  --config configs/development-v3.toml --output /private/new-v3-review
```

Fresh output directory required. Exit 2 means blocked: inspect `readiness.json` and
its specific crossing cases, not just the process exit status of the remote job.
Patient-level reports, adjudication, candidate assignments, predictions and any
checkpoints stay outside Git. Aggregate actual findings belong in STATUS.md after
verified retrieval. The real-data result can remain blocked; synthetic tests do not
clear it. Full five-model training is not authorized by this protocol revision.


## Actual bounded result

See STATUS.md for machine-derived counts/support. The CPU job stopped at 1,800
seconds with 422 detailed comparisons, six probable image-copy links, 14 material
uncertain cases and nine candidate-partition crossings. The report/source/nonce
checks passed; review coverage did not. No experimental split was frozen and no
GPU pilot was submitted. All scheduled-but-unfinished work remains unadjudicated.
Future runtime code now checkpoints screening before detailed comparisons so a
bounded stop does not discard that completed work. This was not available in the
actual source-2da502a execution and is not claimed as retrieved evidence.

The private named-case renderer is executable with `python -m
trustretina.data.review_render`; it checks original image hashes and produces
comparison pages without clearing any case. Render against the attached original
source using the existing CPU workflow. No additional GPU smoke is required.
