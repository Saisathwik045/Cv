# Phase 8 complete — frozen local research demo

PR #14 was already merged at `0e14fa5`; its final CI run `40926892656` passed.
Phase 8 continues from that merged main without training, refitting, split changes,
final-data replay, new GPU jobs, paid resources or external publication.
The five backbones and frozen candidate remain unchanged.
[Phase 7 history](history/STATUS-through-phase7.md) preserves the final evidence.

## Launch and immutable artifacts

```sh
python -m pip install -r requirements/app-cpu.lock
python -m pip install --no-deps --no-build-isolation -e .
# Once, in the existing credentialed Kaggle SDK environment:
python scripts/install_inference_bundle.py --download
python scripts/run_demo.py
```

Workspace-internal address: `http://127.0.0.1:8501`; **not a user-accessible cloud preview**.
The available runtime/tools provide no supported private port-exposure mechanism.
The healthy existing process was reused; no bind/security change or public hosting
was introduced. Browser HTTP/WebSocket/pair interaction checks are internal only.
Use the [local alternative and access diagnosis](DEMO_V8.md#cloud-access-diagnosis-and-supported-alternative).
[Setup and walkthrough](DEMO_V8.md) also gives local-only installation from existing
artifacts. Independent fresh retrieval of the original pinned model/Phase 6
parameters verified every hash; the fresh bundle loaded strictly on CPU with the
Kaggle token removed from the runtime process. App runtime needs neither the
ODIR dataset nor Kaggle credentials. Missing/tampered assets fail explicitly.

- Contract: `a80dce8f27f12645539a0312a3ed96af0b5d10b5d7e1cd522af53b9347409c61`.
- Checkpoint: `6871e49772543f3463f9d30dd6c758257022463bf562e21342fd154d7aa0eaf1`.
- Model storage: `saisathwikregula/trustretina-full-v5-convnext-tiny-c349c706/1`.
- Parameters: `saisathwikregula/trustretina-phase6-analysis-ab40b8f2/1`.
- Bundle: ignored `artifacts/inference-v8`; fresh proof `artifacts/demo-v8/fresh-bundle`.

## Implemented behavior and actual checks

Cached strict ConvNeXt-Tiny loading, CPU/available-CUDA selection, both-eye uploads,
previews/errors, fixed preprocessing/mean-eye sigmoid aggregation/shared temperature,
20 trained-head dropout passes, all eight probabilities/positive labels at p>=0.5,
experimental confidence/variance/technical-quality signals, selectable per-eye
Grad-CAM and downloadable versioned JSON are implemented. Every result remains
**REVIEW REQUIRED**; no reliability percentage, acceptance threshold or clinical
guarantee is introduced. Normalization and stochastic depth stay in evaluation mode.

Uploads are memory/session-only; changing either invalidates prior results. Reset
clears session results/widgets/explanations and session upload/media references.
No patient content is cached with the shared model or logged by the app.
Downloaded/browser copies cannot be revoked; reset is not secure erasure.

The results page reads hash-verified aggregate files, separating all-five validation
comparison from the held-out evaluation. It preserves calibration heterogeneity
(D/H NLL worsened), unsupported head-MC benefit and adverse quality fusion.
Labelwise binary accuracy is not diagnostic accuracy. The holdout is used and
must never tune later parameters. [Full final research report](../results/final-v7/REPORT.md).

- Actual CPU parity: first three prespecified saved validation pairs, maximum raw
  absolute probability difference **2.16e-7**, versus declared atol=1e-5/rtol=1e-4.
- Model state invariant after prediction/repetition/Grad-CAM; strict loading and
  target-D spatial maps verified. Exact repeated-pair MC reproducibility and
  internal frozen-helper variance parity (atol=1e-7) passed.
- Pair inference **0.214–0.274 s**, load **1.020 s**, Grad-CAM **0.311 s**; peak
  verification-process RSS **844.6 MiB** (includes checkpoint/arrays, not isolated
  incremental app memory). No CUDA device was available; app CUDA path unexecuted.
- System Chromium/Playwright browser interaction passed: single-eye blocking,
  real development-pair upload/prediction, eight-probability JSON download, target-D
  Grad-CAM for both eyes, reset, invalid-image error and separate research tables.
  No browser page errors. Public screenshots contain only empty UI/aggregates;
  development-image screenshots/JSON stay in ignored `artifacts/demo-v8`.
- Local full suite: **148 tests run, nine optional skips, all remaining passed**.
  Ten focused app tests passed. Compile, dependency consistency and diff checks pass.
  The focused PR adds locked CPU app CI on Python 3.11/3.12.

Evidence: [verification](../results/demo-v8/verification.json),
[browser checks](../results/demo-v8/ui-verification.json),
[fresh recovery](../results/demo-v8/artifact-recovery.json),
[empty preview](../results/demo-v8/preview-empty.png),
[aggregate preview](../results/demo-v8/preview-results.png).

## Limits and next executable step

Single-pair CPU/runtime dropout masks cannot reproduce saved cohort-wide CUDA
masks bitwise; seeds/scope/pass count/formulas and distribution are unchanged,
and batch context is exported. Uncertainty is head-only; quality is a technical
proxy; heatmaps are not validated lesion localization. Internal cohort, inherited
patient labels, sparse classes and lack of external/clinical validation remain.
There is no automatic clinical acceptance policy. The demo has no authentication
and stays local/private. Browser/framework memory cannot promise secure erasure.

Next: review the focused Phase 8 PR and walk through the local demo. Any external
release, clinical validation or new experiment requires separate scope; do not
retune this pipeline using the now-used final holdout.
