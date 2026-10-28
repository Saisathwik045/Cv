# Automated execution and artifact contract

This document specifies the full later pipeline. Phase 1 implements discover-data,
prepare-data and runner checks (see DATA_PIPELINE.md and GPU_RUNNER.md). Foundation
check-env, check-config and plan also execute today; `plan` returns a declarative five-model inventory. No shell
command below is presented as an implemented TrustRetina runner.

## Single-entry execution

Phase 1 now implements validation and splitting; Phase 2 adds one noninteractive
orchestrator that follows a stage DAG and exits nonzero on failed/missing inputs.
It runs validation → approved patient split → preprocessing/cache → five sequential
model jobs → validation predictions/selection → calibration predictions → calibration
and policy freeze → locked final predictions → evaluation → Grad-CAM → reports.
The final-evaluation stage stays gated until Phase 5 protocol freeze. Each job is
independent and batchable; sequential execution is the default on one GPU. Upload
app consumes a frozen inference bundle without initiating training.

## Validation and persistent manifests

Persist a dataset inventory with workbook SHA256, relative image paths/content
SHA256, patient ID, eye, eight source labels, decode results and exclusions.
Store patient manifests with schema version, dataset fingerprint, partition,
seed, stratification implementation/version and protocol version. Expand both eyes
only after assigning patients. Hash manifests; enforce disjointness, completeness,
no duplicate-content leakage and per-class positive/negative support at every
entry point. Refuse automatic overwrite/re-split when manifests already exist.
Sensitive manifests and all generated artifacts live outside version control.

## Run identity and provenance

Run identity is the hash of canonical resolved experiment settings **plus** dataset
fingerprint, split hash, preprocessing/weights hashes, source commit SHA and seed.
The Phase 0 plan hash covers only configuration and is not a reusable execution
identity. Store portable relative input paths plus runtime resolved paths separately
in future canonical configs to permit moving identical runs between machines.
Require a clean checkout or archive/hash the exact diff. Record command, UTC start/end,
Python/package lock/freeze, hardware, CUDA/cuDNN, determinism flags, source version,
pretrained weight source and experiment/protocol schema versions.

Each stage writes into `artifacts/<experiment>/<backbone>/<seed>/`:

| Artifact | Required content |
| --- | --- |
| config.json, provenance.json | Resolved settings, hashes, code/environment versions |
| state.json | pending/running/interrupted/failed/completed, attempts, stage keys, error |
| history.jsonl | Epoch, losses, validation metrics, learning rate, timings, global step |
| checkpoints/last.pt | Model, optimizer, scheduler, AMP scaler, epoch, step, sampler and all RNG states |
| checkpoints/best.pt | Best validation metric/epoch, model and provenance |
| predictions/<partition>.parquet | Patient/image/eye IDs, ordered labels, logits/probabilities, thresholds, model/split hash |
| reliability/, calibration/ | Quality scores, MC samples or sufficient statistics, fit indices, parameters, policy and hashes |
| metrics/*.json | Definition/version, counts/support, thresholds, undefined reasons, confidence intervals |
| stage-complete.json | Input key, output names/sizes/SHA256, schema and successful validation |

Partition labels must remain inaccessible to selection code. Separate final-evaluation
storage/access from tuning processes; do not place final labels in early report files.

## Resume and interruption

Write artifacts to temporary siblings, flush, then atomic rename; completion marker
is written last after validating outputs. Lock each run to prevent concurrent writers.
On restart verify input keys and every required output checksum/schema before skipping
a completed stage. Missing/corrupt files invalidate completion; do not treat a marker
or filename alone as success. Incompatible config/data/code requires a new run or an
explicit, recorded migration, never silent checkpoint reuse.

Save last checkpoint every epoch and on supported termination signals. Checkpoints
include Python/NumPy/CPU/CUDA RNG, optimizer, scheduler, scaler, sampler, best score
and patience state. Resume at the next completed epoch by default; an interrupted
epoch may repeat, but already completed epochs/stages do not. Specify sampler and
worker seed behavior and test an interrupted-versus-continuous synthetic run before
GPU work. Retain best independently from last. Exit nonzero on failure and store
traceback without credentials; bound retries to transient runner faults.

Kaggle session storage is ephemeral: local last.pt is insufficient across sessions.
Use time-bounded chunks finishing before the session cap, publish checkpoints as
private versioned output artifacts, download/verify and attach the previous output
as the next job input. Abrupt termination may lose unexported progress; this is a
known limitation until durable periodic export is verified. No claim of lossless
cross-session resume is made in Phase 0.

## Results and figures

A future report generator loads validated metrics and prediction files, verifies
matching data/split/protocol hashes, and computes comparisons/figures. Include run
status and artifact provenance; label incomplete comparisons and refuse stale or
incompatible inputs. Generate metric tables, learning curves, class support,
calibration diagrams and risk/coverage curves from artifacts only. Never ship
invented sample results. CI uses synthetic fixtures, with no data or GPU downloads.
