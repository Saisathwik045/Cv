# TrustRetina

The five-backbone v4 development comparison is complete. See the
[actual validation results](results/full-v5/README.md),
[current status and recovery commands](docs/STATUS.md), and
[prespecified training protocol](docs/FULL_EXPERIMENT_V5.md).
Reliability and calibration experiments are next; final evaluation remains untouched.

Automated retinal screening **research**, using original ODIR-5K and authoritative
`data.xlsx` patient labels. Class order: **N, D, G, C, A, H, M, O**.

Retinal image → preprocessing → eight-label classifier → confidence + MC Dropout
uncertainty + image quality → calibration → ACCEPT / REVIEW REQUIRED → Grad-CAM.

Compare EfficientNet-B0, ResNet50, DenseNet121, ConvNeXt-Tiny and EfficientNet-B2.
An image inherits its patient's source labels; this does not establish eye-specific
diagnosis. Both eyes always stay in the same partition. Research output is not a
validated clinical decision system.

## Setup and working commands

Python 3.11 or 3.12; no GPU, dataset, credentials or notebook execution needed for
foundation checks; Phase 1 tests additionally require the data extra. From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[data]"
python -m unittest discover -s tests -v
python -m compileall -q src tests
trustretina check-env
trustretina check-env --network
trustretina check-config --config configs/phase0.toml
trustretina plan --config configs/phase0.toml
```

`check-env` reports observations, including unknown CUDA status when torch is absent.
`--network` makes bounded read-only HTTPS probes. `--require-gpu` exits 1 unless a
PyTorch CUDA device is usable. Configuration errors exit 2. `plan` prints a
**declarative inventory**, marks jobs `not_implemented`, and never starts work.
Dataset validation/splitting, the five-backbone training/evaluation modules and
prediction-derived reports are implemented. The Phase 0 `plan` command remains
declarative; the completed experiment uses the dedicated resumable runner linked
in the current status. Clinical deployment and fitted reliability/calibration
claims are not established.

For a preinstalled managed environment with unavailable package downloads, the
verified offline installation is:

```sh
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
```

This fallback requires setuptools >=68 already installed. A source-only alternative
is `PYTHONPATH=src python -m trustretina check-env`.

Optional dependency groups in `pyproject.toml` separate `data`, `train`, `reports`,
`app`, and `runner`. They are declarations, not a validated GPU dependency lock.
Before Phase 2 choose a compatible torch/torchvision/CUDA combination using the
[PyTorch installer](https://pytorch.org/get-started/locally/), resolve and freeze the
complete runner environment, and archive `pip freeze` with every run.

## Configuration and data

Copy `configs/phase1.toml` to ignored `configs/local.toml` and adjust `[data]` and
`[paths]` for the original dataset. Relative paths resolve from the configuration
file's directory, not the shell directory. Do not put Kaggle mount paths in Python.
Configuration validation does not validate or download the dataset. Run
`trustretina discover-data --config configs/local.toml` and then
`trustretina prepare-data --config configs/local.toml` for read-only dataset
validation and an immutable patient split bundle. See [Data pipeline](docs/DATA_PIPELINE.md).

Keep raw inputs read-only and unmodified. Do not commit data, patient manifests,
images, predictions, weights, secrets or tokens. Existing notebooks, if added,
are reference material; all future execution must use scripts and a single
orchestrator entry point without manual cell execution.

## Project map and current scope

`src/trustretina/` contains the implemented configuration, planning and capability
checks. The data subpackage implements workbook/image validation and immutable patient
manifests; the runner subpackage implements connectivity and smoke checks. Remaining
subpackages reserve ownership for preprocessing, models, training,
evaluation, reliability, calibration, Grad-CAM, inference and the upload app;
their module docstrings explicitly mark future implementation.

- [Phase 0 review and merge](docs/PHASE0_REVIEW.md)
- [Phase 1 data pipeline](docs/DATA_PIPELINE.md)
- [Project context](docs/PROJECT_CONTEXT.md)
- [Roadmap and acceptance criteria](docs/ROADMAP.md)
- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Execution and artifact contract](docs/AUTOMATION_DESIGN.md)
- [GPU runner investigation and required actions](docs/GPU_RUNNER.md)
- [Environment observations](docs/ENVIRONMENT.md)
- [Status and next action](docs/STATUS.md)
- [Git workflow](docs/GIT_WORKFLOW.md)

Mechanical data validation and historical conservative-group split audits are recorded
in [Status](docs/STATUS.md), with patient-level artifacts kept private outside Git.
The published splits are not approved for training/calibration; the giant weak-hash
component and rare-class support require protocol review. The CUDA arithmetic smoke
is verified; no disease metrics, trained checkpoints or
calibration improvements are claimed. `trustretina check-runner` reports observed readiness and
exits nonzero when authentication cannot be verified. Smoke preparation never
submits a job; execution requires the separate `run-runner-smoke` command.


CPU implementation contracts and prospective protocol:
[Phase 2 CPU modules](docs/PHASE2_CPU.md). Install the pinned CPU lock and run the
non-fitting tests described there. Training remains blocked by duplicate/split and
calibration protocol review; no training command is exposed.


Current prospective development workflow: [protocol v3](docs/DEVELOPMENT_V3.md).
`trustretina check-development --config configs/development-v3.toml` checks the
four-role, shared-calibration and bounded-pilot contract without fitting anything.
Actual readiness depends on verified private real-data evidence.


Current contained development release and bounded execution evidence:
[protocol v4](docs/DEVELOPMENT_V4.md) and [actual status](docs/STATUS.md).
The fixed review is complete, v4 private splits are frozen and the sole authorized
B0 pilot passed. Full five-model runs remain outside this execution scope.
