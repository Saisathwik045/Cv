# Frozen research demonstration v8

Phase 8 packages the existing candidate; it performs no training, calibration fit,
threshold/strategy selection, split changes or final-evaluation replay. PR #14
was already merged at `0e14fa5`; its final CI run `40926892656` passed. This phase
starts from that merged main. All five backbone factories remain unchanged.

## Apple-silicon Mac: minimum-effort launch

See [Mac quickstart](MACOS_QUICKSTART.md). Install native Python 3.11/3.12 once,
extract/clone the project, double-click `Setup_TrustRetina.command` once to
restore the checksum-verified private bundle, then double-click
`Start_TrustRetina.command`. It manages a project-local environment and opens
your local browser only after health succeeds. macOS 13+ / ARM64 is targeted;
portable Linux and wheel-metadata checks are recorded, **not native Mac testing**.
The frozen model and REVIEW REQUIRED behavior are unchanged.

## Install and launch on Linux

Use Python 3.11/3.12 on Linux from the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/app-cpu.lock
python -m pip install --no-deps --no-build-isolation -e .
```

Install the private bundle once. On a credentialed machine with the existing
Kaggle SDK/account (the existing runner environment can run this command):

```sh
python scripts/install_inference_bundle.py --download
```

The optional recovery client is `kaggle==2.2.5`; it is deliberately absent from
runtime app dependencies. Existing verified local artifacts can be installed
without network or credentials:

```sh
python scripts/install_inference_bundle.py \
  --checkpoint artifacts/runner/full-v5/convnext_tiny/attempt-1/download/full/best.pt \
  --parameters artifacts/runner/phase6/analysis/download/full
```

Both commands verify the exact contract, checkpoint, temperature, quality
normalizer and private candidate hashes before atomically installing
`artifacts/inference-v8`. An existing bundle is verified rather than overwritten.
Only the pinned checkpoint and three parameter files are downloaded; no dataset,
training job, calibration arrays or final predictions are needed. Copy this
private bundle to another authorized machine and use `--destination` or
`TRUSTRETINA_BUNDLE` for a different location. Keep it outside Git. Missing or
corrupt bundles fail explicitly; no random weights can produce an app result.

```sh
python scripts/run_demo.py
```

On the **same computer running the command**, open `http://127.0.0.1:8501`.
This is a local address, not a cloud preview link. When the command runs in a
remote cloud workspace, opening that address on your phone/laptop points to your
own device, not the workspace. The launcher uses two CPU threads by default and
loopback binding; it does not provision hosting. The app has no Kaggle dependency
or token lookup once installed. The exact CPU dependency lock is separate from
the historical GPU lock. CUDA is selected automatically when an installed
compatible Torch/torchvision runtime reports it available. No CUDA app execution
was available in this workspace; that path has not been reverified here.

## Cloud access diagnosis and supported alternative

The access investigation on 30 October 2026 found the existing process alive at
PID 2447, listening only on workspace `127.0.0.1:8501`. It was reused; no duplicate
server was started. An internal health response is insufficient evidence that a
user's device can reach the app. The earlier “verified private preview” wording
and queued browser card did not establish device-accessible forwarding.

The current managed-environment status reports no configured capabilities, and
the available tool catalog exposes no private web-preview/ingress registration
or port-exposure operation. The cloud-runtime networking reference documents
outbound HTTP; its VPN forwarding reference explicitly excludes inbound
connections. VPN is not configured here. `open_in_codex` opens/queues a browser
tab; it does not register a forwarded port or return a hosted preview URL.
Consequently **no supported device-accessible private preview URL can be issued
from this task**. There is no registered URL whose base path/WebSocket proxy can
be tested. Switching the bind address to `0.0.0.0` would not create that route,
so the healthy listener and default CORS/XSRF protections remain unchanged.
No public tunnel, publication or paid hosting is used.

A fresh workspace-internal Chromium check verified the main page, Streamlit
`/_stcore/stream` WebSocket and received frames, both upload controls, an existing
private development-pair prediction, result download control and reset without
page/WebSocket exceptions or displayed startup/artifact-load errors. This
**does not verify external/device access**. The original detached execution
session is no longer available to this turn; access to its stdout/stderr file
descriptors is denied, so complete server logs could not be inspected. The
private check record is `artifacts/demo-v8/access-diagnosis.json`.

The smallest supported alternative is local execution on your own computer.
Use the dependency installation above, then either copy the verified private
`artifacts/inference-v8` bundle or restore it with the optional authorized Kaggle
SDK. For that one-time recovery, install `kaggle==2.2.5` in the setup environment.
From an already dependency-installed checkout, this single shell command verifies
or installs the bundle and launches locally:

```sh
python scripts/install_inference_bundle.py --download && python scripts/run_demo.py
```

For an already installed bundle, only `python scripts/run_demo.py` is needed.
The browser must run on that same computer. No Kaggle token is required by the
running app. Do not upload private images to an unapproved tunnel or hosting site.

The existing workspace process is left running, but the original execution
session cannot be reattached from this turn. Its observed survival is not a
managed restart or uptime guarantee. It stops when the cloud environment stops;
no permanent hosting or future monitoring is promised. A local launch lives as
long as its terminal/process and computer remain running. If the platform later
provides an authenticated private port-preview feature, register the port through
that feature first, use only its returned URL, and separately verify HTTP,
WebSocket, upload interactions and access control through that URL before sharing
it. Bind to `0.0.0.0` only if that documented mechanism requires it.

## Walkthrough

1. Upload separate left- and right-eye JPEG/PNG files for the same patient. Each
   image must be at most 20 MiB / 25 million pixels. Missing/corrupt images have
   readable errors. One eye is never duplicated to create a pair.
2. Select **Analyze paired images**. Inspect all eight raw/calibrated probabilities,
   the full positive-label list at fixed p >= 0.5, and experimental signals.
   The decision is always **REVIEW REQUIRED**. There is no acceptance threshold,
   clinical diagnosis, or invented reliability percentage.
3. Select an explanation target independently of the positive-label list, then
   generate the two Grad-CAM overlays. They explain the raw mean-eye target score
   on the fixed letterbox canvas; they do not verify lesion localization.
4. Download JSON containing the frozen versions/hashes, probabilities, signals,
   seeds, execution environment and REVIEW REQUIRED. It contains no images,
   filenames or patient identifiers. Reset clears session results, upload widgets,
   explanations and this session's server upload/media references. The model
   cache is shared; patient data is never cached with it. Browser/downloaded copies
   cannot be revoked; memory cleanup is not a secure-erasure guarantee.
5. Open **Research results** for checksum-verified validation comparison and the
   separately labeled held-out tables, confidence intervals and diagrams. This
   includes pooled calibration gains, D/H classwise NLL deterioration, uncertain
   head-MC ranking benefit and worse quality fusion. Labelwise binary accuracy
   is explicitly not diagnostic accuracy. Coverage subsets are retrospective
   research analyses, not a clinical acceptance policy.

## Frozen inference semantics

`InferenceService` verifies contract SHA256
`a80dce8f27f12645539a0312a3ed96af0b5d10b5d7e1cd522af53b9347409c61`
and every bundle file before strict checkpoint loading. The cached model is
ConvNeXt-Tiny epoch 2 with checkpoint SHA256
`6871e49772543f3463f9d30dd6c758257022463bf562e21342fd154d7aa0eaf1`.

Class order N,D,G,C,A,H,M,O; EXIF RGB224 letterbox, ImageNet normalization and
mean of both eye sigmoid probabilities are unchanged. The saved shared
T=0.7226021421263825 acts on logit(clipped mean eye probabilities), followed by
sigmoid. It does not act on mean eye logits. All positive labels are returned,
using p >= 0.5; no softmax or diagnostic argmax is used.

Twenty passes reuse seeds 62026..62045. Only trained `head.0` dropout p=0.2 is
active. Stochastic depth and normalization remain in evaluation mode; parameters
have gradients disabled. Both eyes are averaged within each pass. Raw/calibrated
confidence is min(max(p,1-p)) across labels; uncertainty is the maximum per-label
population variance across passes. Quality uses the three existing canonical
technical features, saved sorted training ECDFs and minimum of the eye scores.
The same existing formulas are used, without fitting. All fixed exploratory
fusion signals are returned; none gates an automatic acceptance decision.

Seeded masks depend on device/runtime and batch shape. A single-pair app request
has the same stochastic-head distribution and aggregation but cannot reproduce
cohort-wide saved CUDA masks bit for bit. JSON records this context. RNG forking
and a process-wide lock prevent concurrent requests from interfering; the
original RNG state and layer modes are restored. No uploaded content is sent to
Kaggle. Original source labels are patient-derived, not verified eye diagnoses.

## Executed verification and recovery

`results/demo-v8/verification.json` records an actual strict CPU checkpoint load,
three prespecified saved validation pairs (canonical PNG cache examples), parity,
repeatability, Grad-CAM, unchanged state, runtime and process memory. Declared
parity tolerances are atol=1e-5, rtol=1e-4; maximum observed error 2.16e-7. Paired
inference took 0.214–0.274 seconds, model loading 1.020 seconds, target-D Grad-CAM
0.311 seconds. Peak process RSS was 844.6 MiB, including checkpoint loading and
verification arrays; it is not an isolated incremental app memory figure.
Internal frozen-helper MC-variance parity uses atol=1e-7, rtol=0; repeated
same-pair CPU MC variance is exact. Saved cohort-wide stochastic-mask bitwise
parity is unavailable for the reasons above. No final sample was accessed.

`artifact-recovery.json` records an independent fresh download from the original
pinned private model/Phase 6 parameter outputs, with every hash checked. Browser
verification runs against the local app using the same private development pair;
its screenshots with fundus images stay in ignored `artifacts/demo-v8`.
Only empty-input and aggregate-results screenshots are permitted in Git.

```sh
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests streamlit_app.py
# Requires the existing private development cache; never downloads or evaluates final data:
python scripts/verify_demo.py
# Optional: pip install -r requirements/app-ui.lock
# Requires system Chromium and the running local preview:
python scripts/verify_demo_ui.py
```

No new GPU job, fitting, training, final metric calculation or paid resource is
part of these checks. Development fixtures and JSON downloads remain private.
This local demo has no authentication and must not be exposed publicly as-is.
