# Clean-environment software reproduction

Observed on 20 November 2026: **PASS — 380 CPU tests, zero skips, failures or
errors** in a fresh Python 3.12.14 Linux virtual environment with system site
packages disabled. The package was freshly installed editable from a separate
421-file source copy of main `6fe5a0f83429fa9182da0b6c3117d9df457fa53e`
plus this phase's test/documentation changes. This is an isolated source copy,
not a new OS installation or a claim that unmerged changes are already on main.
Final reporting prose was completed after the run; test/runtime source was unchanged.
The exact source inventory, install/check log and aggregate receipt are retained
privately; no private research data is included in that source copy.

## Observed checks

| Check | Result |
|---|---|
| Fresh venv, no system site packages | PASS |
| Combined unchanged report/app locks and optional photographic packages | Installed successfully from package registries; local download caches may be used |
| Explicit setuptools/wheel and editable package install | PASS; no reuse of the pre-existing project virtualenv |
| `pip check` | PASS |
| Phase0/phase1 config, declarative plan, v3/v4 development contract checks | PASS; no training or job execution |
| Environment report and compilation | PASS; no GPU access inferred |
| Full unittest discovery | 380 run, zero skipped/failed/errors |
| Test network and private-storage access | Socket connections and private research-storage file opens blocked in the test process |

The full suite includes existing generated toy-model/optimizer fixtures and public
historical aggregate display. These are software tests, not private-cohort access,
real-model inference, real-data fitting, repeated training or historical-final replay.
The new [robustness cases](DEMO_ROBUSTNESS_AUDIT.md) ran in this full environment.
No checkpoint, patient image, label or private manifest was installed or accessed.

## Reproduction recipe

Use a fresh checkout of the reviewed submission revision, then follow
[the submission checklist](FINAL_SUBMISSION_CHECKLIST.md). Explicit build tools
are necessary for the `--no-build-isolation` editable-install recipe, especially
when a fresh Python venv does not include setuptools or wheel.

```sh
python3.12 -m venv .venv-submission
. .venv-submission/bin/activate
python -m pip install setuptools==78.1.0 wheel==0.48.0
python -m pip install -r requirements/full-v5-reports.lock -r requirements/app-cpu.lock
python -m pip install opencv-python-headless==4.11.0.86 scikit-image==0.25.2
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -B -m unittest discover -s tests -v
python -m compileall -q src scripts tests dashboard streamlit_app.py
trustretina check-config --config configs/phase0.toml
trustretina check-config --config configs/phase1.toml
trustretina plan --config configs/phase0.toml
trustretina check-development --config configs/development-v3.toml
trustretina check-contained-development --config configs/development-v4.toml
trustretina check-env
git diff --check
```

Run public-only tests in an environment without credentials/private datasets.
The observed audit additionally used a test-process socket/file-open guard; the
plain commands above do not themselves install an OS sandbox or block networking.
Do not run any separately gated real benchmark/recovery/analysis command as part
of this software reproduction. Missing private assets must remain missing.

## Dependency evidence and limits

Observed versions: torch 2.7.1+cpu, torchvision 0.22.1+cpu, NumPy 2.2.6,
pandas 2.2.3, Pillow 11.3.0, Streamlit 1.50.0, matplotlib 3.10.8,
setuptools 78.1.0, wheel 0.48.0, OpenCV headless 4.11.0.86 and
scikit-image 0.25.2. Core locks were not modified:

| File | SHA256 |
|---|---|
| `requirements/app-cpu.lock` | `d9e1407755dfdf292c1f18b99a7ddb6dc42828212856ee61a1db36a19c964bea` |
| `requirements/full-v5-reports.lock` | `df4d9305e751adaced3f87d4db2ea99052b43330e8184f9c8839e977dc81c678` |
| `requirements/phase2-cpu.lock` | `51c537db0b5d0e9e55fc17896e6f46774dfc3f943b86114c55c552dc82351b14` |

These are version pins, not package-file supply-chain hashes. Optional package
transitive dependencies can resolve differently in future; the private install
log records the observed set. Success on this Linux/Python 3.12 host does not
prove native Mac/Windows behavior, all future dependency combinations, external
cloud viewing, offline package availability or bit-for-bit scientific reproduction.
Python 3.11/3.12 are additionally checked by the PR's CPU CI matrix; native Mac
rehearsal is optional for the student's chosen cloud-only demonstration.
