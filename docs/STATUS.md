# Phase 8 Mac packaging — native execution pending

PRs #15/#16 were already merged; their final CI runs `40999463918` and
`41002913956` passed. Packaging continues from latest main `b0e6f9c`.
[Apple-silicon quickstart](MACOS_QUICKSTART.md) gives the minimum local steps:
install native Python 3.11/3.12, clone/extract, double-click
`Setup_TrustRetina.command` once, then `Start_TrustRetina.command`. Targets
macOS 13+ ARM64 CPU execution. No further cloud-preview work or tunnel is used.

The launcher locates its checkout, verifies all frozen bundle hashes, manages
`.venv-mac` with an exact Mac lock/installed-version checks, verifies native imports
on installation, reuses its healthy identified server, waits for loopback health
and then opens the browser. Startup locks prevent duplicate launch races.
Changed app-entry/dependency hashes require restarting the original instance.
The owning Terminal controls lifecycle; Control-C/hangup stops its child and
cleans its receipt. No daemon, public hosting or permanent uptime is promised.

One-time artifact setup has a separate `.venv-artifacts-mac` / exact recovery lock,
uses the original pinned jobs, accepts a hidden token without saving it, or reuses
secure existing local credentials. Copying `artifacts/inference-v8` privately skips
credential setup. The running app receives no Kaggle environment variables and
needs neither Kaggle SDK/credentials nor ODIR. No random-weight fallback exists.
Lifecycle logs are private, controlled metadata only; raw subprocess/server
output, credential values and uploaded contents are not persisted.

PyPI wheel/Requires-Python/conditional-dependency metadata checks passed for both
exact locks on macOS 13 ARM64 / CPython 3.11 and 3.12. This is not a native install.
Twenty-one focused portable launcher tests pass. The full local Linux suite ran
169 tests: 160 passed and nine optional checks skipped. A real Linux Streamlit fixture
verified paths with spaces, HTTP health, process identity, second-launch reuse of
the same PID, and Control-C cleanup. It performs no model inference/evaluation.
Evidence: [wheel metadata](../results/demo-v8/macos-wheel-metadata.json),
[portable lifecycle](../results/demo-v8/macos-portable-verification.json).

**Not executed on macOS:** Finder/Gatekeeper, native environment installation,
default-browser opening, strict checkpoint execution, latency or numerical parity.
No Mac runner or paid resource was provisioned. The frozen model, preprocessing,
mean-eye aggregation, temperature, 20 head passes and REVIEW REQUIRED behavior
are unchanged. No training, recalibration or real-data evaluation was repeated.
Next: review the packaging PR, then perform the documented first run on the
user's Apple-silicon Mac. The linked historical Phase 8 evidence remains Linux evidence.


[Prior Linux implementation and access diagnosis](history/STATUS-through-phase8-linux.md)
preserves the earlier evidence and private recovery commands.
