# Phase 1 dataset pipeline

The implemented pipeline performs discovery, original-workbook parsing, image
validation, deterministic patient grouping/stratification and persistent manifests.
**Only synthetic fixtures have been validated in this session. Real ODIR data is
absent; no real-data counts, splits, support or disease results are claimed.**

## Execute

Install `python -m pip install -e ".[data]"`. Copy `configs/phase1.toml` to ignored
`configs/local.toml`. Set data.root to the original labeled cohort directory;
data.labels and data.images are relative to that root. Then run:

```sh
trustretina discover-data --config configs/local.toml
trustretina prepare-data --config configs/local.toml
```

Discovery lists candidate `data.xlsx` files and matching image directories; it
never silently chooses between candidates. Update the exact relative labels/images
paths in local config based on discovery. If several sheets contain all required
columns, specify data.sheet. Do not repoint the configuration to a processed CSV
or merge official test cohorts into training. The default bundle is
`artifacts/data/phase1`; `--output` can select another immutable bundle directory.
Neither command downloads data, trains a model or runs a notebook.

## Validation and source semantics

Read `data.xlsx` with openpyxl in read-only mode, with formula expressions visible
and rejected in required fields. Require unique ID, Left-Fundus, Right-Fundus and
each of N,D,G,C,A,H,M,O exactly once. Extra diagnostic/demographic columns do not
become targets. Labels must be numeric binary, nonmissing and noncontradictory;
all-zero targets and N together with disease labels require adjudication. No
automatic relabeling, exclusions or destructive repair occurs.

Canonicalize numeric IDs to integer strings (including numeric text); reject ID
collisions. Original-style image stems must be `<canonical-ID>_left` and
`<canonical-ID>_right`. Both files must resolve under the configured image directory,
exist and decode. Escaping paths/symlinks, duplicate eye references, corrupt images,
and decompression-bomb warnings fail validation. **The eight targets belong to the
patient and are copied honestly to both eyes; they are not verified eye diagnoses.**

Hash workbook and referenced files with SHA256 and check for mutation during reads.
Record decoded, EXIF-oriented RGB pixel hashes, dimensions and 64-bit dHash.
Duplicate file/pixel content and dHash distance <= data.near_duplicate_hamming
(default 2) are fully enumerated. The default for unspecified duplicate_policy is
`reject`. Phase 1 explicitly selects `conservative_patient_components_v1`: exact
matches and unresolved perceptual candidates are retained and connected patient
IDs are co-grouped transitively before splitting. No image is deleted or label
changed. Perceptual similarity does not prove duplication; paired 64x64 RGB
comparisons and contrast/entropy are descriptive evidence, not clinical adjudication.
Conflicting patient annotations remain visible in the private adjudication artifact.
See the versioned policy in EXPERIMENT_PROTOCOL.md. Pairwise screening is quadratic
and intended for ODIR-scale data. No candidate is dismissed or threshold changed
based on disease/model results.

Only workbook-referenced eyes enter the inventory. Extra/unlabeled official test
images stay outside all labeled manifests and cannot produce test metrics. This
phase handles one explicitly selected labeled cohort; it does not infer official
cohort boundaries from filenames or directory names. Select the original cohort
explicitly and preserve any external boundaries when mounting it.

## Stratification and persistence

Use `trustretina-component-rarest-label-v2`, a local deterministic component-level
multilabel algorithm: allocate indivisible connected patient groups for rarest remaining labels to the largest unfilled
per-partition label target, use remaining partition capacity for ties, and use a
seeded patient ordering for deterministic ties. Patient-count targets use largest-remainder
rounding of .65/.15/.10/.10. Whole groups may change realized counts. Never expand a patient into eyes before assigning it.
This replaces the optional external iterative-stratification dependency; no seed
search, final outcome tuning or optimization over evaluation results is performed.

Check global feasibility first and actual positive/negative patient support for all
eight classes in every partition afterwards. A greedy split may fail the support
floor even if a more elaborate allocation exists. Stop and review the protocol;
never silently retry seeds or publish an inadequately supported split. The initial
minimum of one positive/negative is only a feasibility floor, not statistical power.
Calibration's later internal fit/policy split still needs its own support audit.

The published directory contains inventory.json, adjudication.json, manifest.json, four separate
partition JSON files, provenance.json and complete.json. Artifacts include relative
paths, labels, paired eyes, data kind, workbook/file hashes, dataset fingerprint,
algorithm/version, split settings, seed, support, implementation-source hashes,
source commit and dependency versions. The fingerprint excludes mount locations,
so unchanged data can move without changing its identity. There are no timestamps
in deterministic output. A completion marker stores every artifact checksum.

Publish using an exclusive lock, staging directory and rename, with completion
written last. Reruns revalidate raw data, compare input/implementation keys, verify
all checksums and audit grouping, labels, completeness and support before reuse.
Existing bundles are never overwritten. A crash before publication leaves no usable
bundle; a stale `.lock` requires confirming no writer is active before removal.
Do not store patient manifests in Git. Later training must load only its permitted
partitions; final-evaluation JSON contains labels and must remain access-isolated
from selection code. This phase does not claim filesystem access control alone
prevents all possible label misuse.

## Automated synthetic verification

`python -m unittest discover -s tests -v` creates tiny random-pixel images and XLSX
files in temporary directories. These are explicitly marked with data.kind =
synthetic_fixture and a TRUSTRETINA_SYNTHETIC_FIXTURE worksheet. The validator
refuses to process those marked fixtures as original_odir. No binary fixtures,
patient manifests or model outputs are committed.

Tests cover label parsing/multi-label preservation, ID/eye integrity, missing and
corrupt images, exact/perceptual duplicates, path escapes, ambiguous sheets,
patient leakage, support failure, deterministic publication/reuse, raw immutability,
artifact corruption and settings changes. Passing these tests establishes code
behavior on synthetic inputs, not ODIR provenance or real-data validation.
