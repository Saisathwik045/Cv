# V15 software execution readiness

V15 implements CPU analysis of hash-verified V14 saved arrays, with synthetic-only
verification in this phase. [Prospective protocol](CALIBRATION_RELIABILITY_V15.md)
defines the exact calculations, gates and limitations. This is software readiness,
not real-data execution readiness or a result report.

The offline package lives in `scripts/trustretina_v15/`, outside V14's frozen
`src/` inventory. Its CLI configures imports; direct library callers use
`PYTHONPATH=scripts:src`. The V14 controller's original source/config/protocol and
orchestration hashes still verify without changing its freeze records.

- Use the existing Python 3.12 CPU virtual environment. The installed
  `requirements/phase2-cpu.lock`, `app-cpu.lock` and `full-v5-reports.lock` match;
  no new dependency is required. Numeric analysis needs NumPy/SciPy/scikit-learn;
  optional PNG export requires Matplotlib. No Torch import is needed by V15.
- Public V14 registry and frozen-plan hashes are pinned by `v15_inputs.py`.
  `readiness` checks only those public files; it does not claim private artifact
  availability. Both recorded outputs are fixed, completed V14 jobs, never
  resubmitted by V15.
- Synthetic fixtures exercise both saved-array candidates, separate scalar fits,
  class/count metrics, quality/MC alignment, paired component intervals,
  reliability/risk plots and non-overwrite outputs. They use opaque fake checkpoint
  bytes; no real checkpoint is loaded, downloaded or hashed in this phase.
- Real execution remains blocked on separate explicit permission, a reviewed
  manifest bound to that permission, permitted saved artifact recovery and
  original role-specific manifest checks. Derived calibration labels must come
  from their original v4 manifest, not GPU output (V14 calibration output has no
  labels). TRAIN derived quality rows must omit labels.
- No final/external cohort or inference input is supported here. A later
  independent evaluation requires its own prospective cohort/endpoint/access
  protocol. BRSET and all-eight compatibility blockers persist.
- Optional OpenCV/scikit-image photographic tests and Kaggle recovery tooling
  were missing in the restored environment. They do not gate this offline numeric
  software; nine existing photographic tests remain skipped locally. Remote CI
  has its existing dedicated duplicate-evidence job.
- Keep generated normalizers, fits, diagrams and signals in private scratch.
  Do not commit them, reuse the demo T, modify historical thresholds or claim
  independent improvement from either calibration or reused validation.

Run the commands in the protocol, all supported CPU tests, compilation,
`python -m pip check`, foundation configuration/plan checks and `git diff --check`.
Record actual local/CI outcomes in STATUS.md and the PR; do not represent expected
CI or synthetic values as observed real V15 evidence.
