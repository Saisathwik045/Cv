# Duplicate protocol review v2 — diagnostic, not approved for training

PR #7 was merged for reusable validation, staged execution and grouping primitives
only, after 62 tests and Python 3.11/3.12 CI passed. Its default is now `reject`;
`conservative_patient_components_v1` remains an explicit historical sensitivity
mode. Historical private manifests are preserved byte-for-byte. No replacement
experimental splits have been approved or published.

## What created the giant component

The original detector hashes the whole EXIF-oriented RGB image after grayscale
conversion and LANCZOS reduction to 9×8, taking 64 horizontal differences. Hamming
≤2 flagged 232,790 image pairs (19,609 at distance 0; 70,537 at 1; 142,644 at 2).
There were **zero raw-byte or canonical RGB pixel-exact matches**. These form
203,002 unique cross-patient relationships. Blanket union of all candidate edges
produced 737 components, including a 2,638-patient component. An independent SciPy
sparse connected-component computation reproduces the original union-find groups.
No union-find corruption or dHash implementation mismatch was observed.

This is weak-hash over-chaining, not proof of a duplicate chain of 2,638 patients.
Shared circular fundus geometry, black surround, coarse intensity structure and
feature-poor photographs make a whole-frame 64-bit hash poorly specific here.
Technical previews show clearly different foreground structures even at Hamming 0–2.
Background inflation is observed: the two independently supported false positives
have full-frame SSIM .857/.864 but foreground SSIM .663/.673. This is descriptive
support, not a quantified attribution of every edge to the border. Sum of marginal
bit entropies is 42.35; that is NOT joint entropy or an independent-bit count. No bit
is ≥99% constant. Genuine copy relationships also exist; the graph cannot simply
be discarded or all its links cleared.

There are 239 patient-relationship bridges, 238 also individual-image bridges.
All 239 relationship bridges were sampled (one representative image edge per
relationship; one bridge has parallel image edges), together with the earlier
70-edge technical sample: 302 distinct candidates. Another 64 non-candidates
(32 Hamming 3/4; 32 ≥12, content-hash ordered) check rejected candidates. Selection
used no class labels, model results, seed search or desired split sizes.

## Independent photographic evidence

Private Kaggle CPU execution, dataset version 2, source commit
`1c2c7b984e6995c989d47bd76249ab4133c8646f`, kernel version
`trustretina-independent-photo-evidence-a3d58de4/1`, completed with GPU disabled,
no internet inside the job, no training and no calibration fitting. Raw/canonical
pixel hashes and all independently recomputed dHashes matched. Receipt nonce,
source script hash, report checksum and every comparison-page checksum verified.

Policy `photo-evidence-v2` was fixed against synthetic controls before reading
real-output classifications. Foreground mask/crop removes the black surround;
fixed 0–255 normalization, 256-pixel comparison, mutual SIFT matches (ratio .75),
RANSAC (3 pixels), spatial coverage ≥.15, ≥12 inliers, inlier fraction ≥.85,
overlap ≥.70, aligned foreground SSIM ≥.95 and normalized MAE ≤.02 jointly support
probable copies. Exact raw/pixel identity confirms a duplicate. Full numeric policy
and actual package versions are in the private report. Numeric tolerances describe
a limited photographic-transform scope, not clinical identity or empirical error
rates. No tolerances were chosen for split sizes or downstream performance.

Resize/JPEG90, mild exposure and black-padding positive controls passed. A
low-information negative remained unresolved. Of 366 evaluated pairs, 204 had a
low-information/insufficient-feature flag; that includes <12 detected features at
this resolution and is NOT a clinical ungradability assessment. The conservative
verifier has limited recall; lack of matching features cannot clear an edge.

| Scope | Confirmed exact | Probable copy | Supported false positive | Unresolved |
|---|---:|---:|---:|---:|
| 302 sampled candidates | 0 | 1 | 2 | 299 |
| 239 bridge representatives (subset) | 0 | 1 | 0 | 238 |
| 64 sampled rejected pairs | 0 | 0 | 0 | 64 |
| All 232,790 candidates, unreviewed retained | 0 | 1 | 2 | 232,787 |

The probable pair has 111/111 feature inliers, .897 spatial coverage, aligned SSIM
1.0 and normalized MAE 0.0, despite differing original file/pixel hashes. Two eyes
from the same two other IDs are supported false-positive hash matches: visibly
different vessel layouts plus weak independent feature evidence. Representative
bridge pages include quite different structures, severe exposure/background
similarities and probable copies. Rejected previews include distinct layouts and
some visually similar candidates; their inconclusive verification does not prove
the detector has no false negatives. No clinical adjudication was performed.

Available workbook metadata was checked for representative pairs. The probable
link has matching age/sex, matching left keywords and patient label vectors, but
different right-eye keywords. A different visually similar unresolved pair has
conflicting age/sex despite agreeing disease vectors/keywords. Both observations
are recorded privately; neither permits clinical identity or correction of labels.
Of all weak candidates, 186,139 have different patient annotation vectors; this is
not 186,139 verified annotation errors. The one supported probable link has no
patient-vector conflict. Conflicting linked annotations must remain visible and
must never be majority-voted or relabeled silently.

## Reproducible containment and sensitivity

`review_policy.py` separates four categories. Confirmed and strongly supported
probable links form transitive patient components; both eyes always follow the
patient. Unreviewed or inconclusive candidates remain explicit unresolved records.
Evidence-supported false positives create no link under the declared copy scope.
All unresolved endpoints are quarantined, closed over confirmed/probable groups.
Quarantine means retained in immutable raw storage, excluded from an experimental
cohort by a versioned private manifest; it never deletes images or changes labels.
This is a diagnostic option, not an approved alternative cohort.

Sensitivity results, holding evidence fixed:

- Exact-only: 3,500 singleton components; unsafe as an approved default because it
  ignores a supported probable copy and material unresolved evidence.
- Exact + supported probable: 3,499 components (3,498 singletons, one size 2).
- All weak candidates: 737 components; largest 2,638, next 44, 14 and 9.
- Explicit unresolved quarantine: 2,807 patients quarantined, including the size-2
  probable-copy group; 693 eligible singleton patients. This dramatically changes
  population composition and cannot silently replace the full cohort.

| Class | All + / − | Quarantine + / − | Eligible + / − |
|---|---:|---:|---:|
| N | 1140 / 2360 | 891 / 1916 | 249 / 444 |
| D | 1128 / 2372 | 968 / 1839 | 160 / 533 |
| G | 215 / 3285 | 138 / 2669 | 77 / 616 |
| C | 212 / 3288 | 167 / 2640 | 45 / 648 |
| A | 164 / 3336 | 131 / 2676 | 33 / 660 |
| H | 103 / 3397 | 92 / 2715 | 11 / 682 |
| M | 174 / 3326 | 104 / 2703 | 70 / 623 |
| O | 979 / 2521 | 813 / 1994 | 166 / 527 |

The eligible H count (11 total) cannot support two independent calibration roles
with even a 20-positive operational floor. Therefore no seed search, forced split
or publication was performed. Neither the exact-only nor all-weak-link sensitivity
extreme resolves the scientific uncertainty.

## Preserved historical partition audit

These are historical counts, **not approved experimental partitions**. Entries
are positive/negative patients, in N,D,G,C,A,H,M,O order.

| Role | Patients | Components | Positive/negative by class |
|---|---:|---:|---|
| Train | 2696 | 11 | 843/1853, 945/1751, 134/2562, 152/2544, 128/2568, 97/2599, 101/2595, 784/1912 |
| Validation | 385 | 335 | 137/248, 98/287, 34/351, 27/358, 18/367, 4/381, 30/355, 97/288 |
| Calibration | 210 | 193 | 80/130, 43/167, 23/187, 17/193, 9/201, 1/209, 22/188, 49/161 |
| Final evaluation | 209 | 198 | 80/129, 42/167, 24/185, 16/193, 9/200, 1/208, 21/188, 49/160 |

The giant component forces train to ≥75.37%, incompatible with a 65% train target;
actual train is 77.03%. Patient and historical candidate-component separation
passed mechanically. That does not make the experimental protocol valid. H=1
calibration cannot justify class-specific calibration or disjoint supported
calibration subdivisions; H=1 final evaluation gives extremely imprecise sensitivity
(e.g. one success has two-sided exact 95% lower bound .025).

## Private artifacts and reproduction

Patient-level pairs, metadata, comparison images, inventories and manifests remain
under ignored `artifacts/runner`, never Git. The private adjudication references
every original pair by index bound to the immutable inventory digest, storing the
category, evidence, decision and reason; no source inventory is overwritten.

```sh
PYTHONPATH=src python scripts/review_duplicate_evidence.py \
  --inventory /private/inventory.json --inventory-sha256 TRUSTED_INVENTORY_SHA \
  --report /private/independent-report.json --report-sha256 TRUSTED_REPORT_SHA \
  --output /private/new-review-directory
```

A fresh output directory is required. Use verified runner retrieval first; this
script checks input hashes, evidence binding and CPU/no-fitting scope but cannot
independently authenticate a remote account receipt. Private evidence includes
submission/verification receipts, source snapshots, execution log, preregistration,
31 comparison pages, graph/bridge census, adjudication, sensitivity/support summary
and a checksum journal. Earlier manifest versions remain unchanged.

Next executable data phase: broaden independent photographic verification and
human technical review, including weak/bridge links and detector false negatives;
validate its recall/specificity under the declared transforms. Freeze a revised
cohort, class-support/power plan and protocol before any deterministic split release.
Do not accept the current historical or quarantine partitions for training.
