# TrustRetina

The dashboard has two pages:

- **Paired-eye prediction** — upload both eyes, view eight-label probabilities,
  compare raw/calibrated probability bars against the 0.5 threshold, view
  confidence, uncertainty, quality, paired Grad-CAM and download the result JSON.
- **Research results** — recorded model comparisons, held-out evaluation,
  per-label confusion matrices, reliability figures, limitations and the
  downloadable research report. Matrices describe the recorded evaluation,
  not unlabeled uploads.

[Presentation verification](docs/PRESENTATION_VERIFICATION.md) ·
[Private Streamlit Community Cloud setup](docs/STREAMLIT_COMMUNITY_CLOUD.md).
Codespaces is optional. Simplifying navigation does not resolve the documented
Community Cloud dependency/XSRF readiness limitations or establish clinical validity.

[Final technical/submission audit](docs/DEMO_ROBUSTNESS_AUDIT.md) ·
[Clean reproduction](docs/CLEAN_REPRODUCTION.md) ·
[Rights, ethics and supervisor sign-off](docs/SUBMISSION_GOVERNANCE_REVIEW.md) ·
[Private backup readiness](docs/PRIVATE_PRESERVATION_READINESS.md).

[Measured CPU performance](docs/CPU_PERFORMANCE.md): original frozen backend,
generated non-patient inputs, observed timing/memory and reproduction boundaries.
The development architecture is implemented; independent clinical validation is incomplete.

[Examiner guide and proposed assessment rubric](docs/EXAMINER_GUIDE.md) ·
[Research evidence and viva defence](docs/ACADEMIC_RESEARCH_DEFENCE.md) ·
[All-eight-class development comparison](docs/ALL_EIGHT_CLASS_COMPARISON.md).
These are supporting documents. The former study/viva, guided synthetic demo,
cloud readiness, examiner rehearsal and preprocessing inspector pages are no
longer part of the dashboard. Their supporting modules and historical evidence
remain in source for offline verification; there are no alternate dashboard routes.

[Final academic report](docs/FINAL_RESEARCH_REPORT.md) ·
[Phase audit](docs/PHASE_COMPLETION_AUDIT.md) ·
[Submission/reproduction checklist](docs/FINAL_SUBMISSION_CHECKLIST.md) ·
[Presentation outline](docs/FINAL_PRESENTATION_OUTLINE.md).
The development project through V15 is documented for academic review; author-specific
formatting/sign-off, independent validation, native Mac verification and durable
private backup remain explicit outstanding work.

[Real V15 CPU calibration/reliability analysis is complete](results/calibration-v15/RESULTS.md).
Shared temperatures are **1.105496125613 at 320 px** and **1.002532561130 at 224 px**.
NLL/Brier decreased slightly, but ten-bin ECE worsened in both cohorts at both
resolutions; classification decisions stayed unchanged and quality fusion worsened
risk-coverage AURC. Calibration diagnostics are in sample; validation was reused.
**Independent validation is incomplete; all decisions remain REVIEW REQUIRED.**
The [prospective protocol](docs/CALIBRATION_RELIABILITY_V15.md), frozen models,
thresholds and original demo remain unchanged. The earlier
[software readiness](docs/V15_EXECUTION_READINESS.md) records its historical scope.

[V14 two-fit execution is complete](results/candidate-v14/RESULTS.md):320/224 validation macro AUROC 0.8596/0.8480 and macro F1 0.5371/0.5332, with important C/A/G/H regressions. These are reused-validation development results, not independent confirmation. Two free Kaggle sessions used 3826.407 quota seconds; original demo, calibration, thresholds and historical evaluation are unchanged.

[Exact frozen inputs/protocol](results/candidate-v14/CPU_READINESS.md), [artifact recovery](results/candidate-v14/verified-artifact-registry.json), [status](docs/STATUS.md). No further GPU job is authorized by completion; BRSET/all-eight independent confirmation remains blocked.

All six authorized [v13 matched resolution fits](results/resolution-v13/RESULTS.md)
completed using **1.924 free Kaggle GPU-quota hours**, within the six-hour/six-session cap.
Mean outer macro AUROC was **0.83273 at224 → 0.85108 at320**; macro F1
**0.53758 → 0.57156**. H/O recall gains came with more false positives;
H F1 declined. [All-eight results, conditional intervals and plots](results/resolution-v13/DISCUSSION.md)
retain those harms.320 is nominated for future research only, without independent
confirmation, full-data refit or demo replacement. The original frozen demo,
historical evaluation, calibration and thresholds are unchanged.
[Status and recovery](docs/STATUS.md) · [Verified inputs and historical preparation](results/resolution-v13/README.md).

[Consolidated development audit](results/development-audit-v12/REPORT.md) is complete.
Square-root weighting did not establish improvement; retain ratio BCE as the
development reference and preserve the frozen demo. Its proposed resolution ablation
was subsequently prepared and authorized as the bounded v13 study above.

All six authorized [training-only nested loss fits](results/nested-v11/RESULTS.md)
completed using 1.143 free Kaggle GPU-quota hours within the four-hour cap.
The paired comparison retains rare-class and false-positive/false-negative findings;
it does not independently confirm improvement.
[Current status and recovery](docs/STATUS.md) track actual jobs and artifact checks.
The frozen demo, calibration and historical evaluation remain unchanged.

[CPU-only exploratory threshold analysis](results/threshold-v10/RESULTS.md) is complete:
group-aware held-out threshold fitting reduced misses but increased false positives;
H did not improve. This reused-validation study does not provide independent
confirmation. The original demo and historical evaluation remain unchanged.

All four approved [controlled development experiments](docs/CONTROLLED_TRAINING_V9.md)
are complete within 1.338 free Kaggle quota hours, including resume.
[Verified comparison, per-class FP/FN/F1 and learning curves](results/round2-v9/RESULTS.md)
retain positive and negative findings. These reuse validation; the frozen demo,
calibration and historical holdout remain unchanged.
[Current status and recovery](docs/STATUS.md).

## Apple-silicon Mac

[Mac quickstart](docs/MACOS_QUICKSTART.md): install native Python 3.11/3.12 once,
download/extract this checkout, double-click **Setup_TrustRetina.command** once,
then **Start_TrustRetina.command**. The launcher installs/reuses locked app
dependencies, verifies the private bundle and opens the local browser after health
succeeds. Targets macOS 13+; native Mac execution is not yet verified.

## Research demo — Phase 8

The complete local Streamlit demo packages the unchanged frozen ConvNeXt-Tiny
pipeline. Both eyes are required; all predictions remain **REVIEW REQUIRED**.
It includes raw/calibrated probabilities, trained-head MC signals, technical
quality, per-eye Grad-CAM, JSON download and separate validation/held-out tables.

```sh
python -m pip install -r requirements/app-cpu.lock
python -m pip install --no-deps --no-build-isolation -e .
# Once, using the existing credentialed Kaggle SDK environment:
python scripts/install_inference_bundle.py --download
# App runtime requires neither Kaggle credentials nor the dataset:
python scripts/run_demo.py
```

Local browser on the same computer: `http://127.0.0.1:8501` (not a cloud preview).
This cloud task has no supported private port-exposure mechanism. See
[installation, access limitations and walkthrough](docs/DEMO_V8.md)
for local artifact installation, exact verification evidence and limitations.
No training, fitting, final evaluation replay or public hosting occurs.


The frozen Phase 7 held-out evaluation is complete. See the
[generated research report](results/final-v7/REPORT.md) for actual metrics,
intervals, negative findings and limitations. No automatic acceptance policy is
validated. Phase 8 now packages the unchanged pipeline into the local research demo.

Phase 6 reliability and shared-temperature calibration are complete. See the
[development findings and limitations](results/reliability-v6/README.md) and
[frozen research candidate](results/reliability-v6/candidate-contract.json).
Automatic acceptance remains unsupported; Phase 7 now records the authorized final evaluation.

The five-backbone v4 development comparison is complete. See the
[actual validation results](results/full-v5/README.md),
[current status and recovery commands](docs/STATUS.md), and
[prespecified training protocol](docs/FULL_EXPERIMENT_V5.md).
The final evaluation is complete; do not tune against this now-used holdout.

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
in the current status. Clinical deployment and independent reliability/calibration
improvement are not established; V15 reports development-only fitting diagnostics.

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

`src/trustretina/` implements configuration, data validation/containment,
preprocessing, the five classifiers, guarded training/resume, evaluation,
calibration/reliability, Grad-CAM, inference and Streamlit session handling.
Dedicated controllers implement the completed historical experiments. The
[phase audit](docs/PHASE_COMPLETION_AUDIT.md) separates implemented software,
completed development execution and incomplete independent/clinical validation.

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

Data validation and superseded early weak-hash grouping are preserved in the
[historical status](docs/history/STATUS-through-phase4.md). Later approved v4
containment supported the completed five-model comparison and later studies.
Early blocked-readiness text describes its original phase, not the current
absence of models or results. Patient-level artifacts remain outside Git.

The [submission checklist](docs/FINAL_SUBMISSION_CHECKLIST.md) gives current
software verification commands and distinguishes app, reporting and recovery
requirements. Historical [v3](docs/DEVELOPMENT_V3.md)/[v4](docs/DEVELOPMENT_V4.md)
protocols remain unchanged. Readiness checks do not authorize any new job,
fitting, private recovery, inference or clinical deployment.
