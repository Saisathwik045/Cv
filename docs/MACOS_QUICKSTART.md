# TrustRetina on your Apple-silicon Mac

The launcher targets **macOS 13+**, **native ARM64 Python 3.11 or 3.12** and CPU
inference. It preserves the frozen ConvNeXt-Tiny pipeline and always returns
**REVIEW REQUIRED**. No dataset, GPU service, training or calibration fit is used.
Intel Macs, Rosetta Python and Python 3.13+ are not supported by this package.

## Minimum steps

1. **Install compatible Python once**, if you do not already have it. If you use
   Homebrew, run `brew install python@3.12`. Otherwise use a compatible signed
   Python 3.12 macOS universal2 installer from
   [python.org](https://www.python.org/downloads/macos/). Prefer a maintained
   Python patch release. The launcher finds native Homebrew/python.org locations
   even when Finder has a limited PATH; it never installs Homebrew/Python itself.
2. **Download this repository** to a normal writable folder, not inside a ZIP or
   read-only disk image. Use GitHub **Code → Download ZIP**, then extract it, or
   clone it with Git. While the packaging PR is open, select its
   `phase/08-macos-launcher` branch first. Keep the complete folder structure.
3. **Double-click `Setup_TrustRetina.command` once.** It creates a separate small
   recovery environment, installs its exact pinned dependencies, and prompts for
   a Kaggle API token without showing or saving it. Get your token from your
   authorized account's **Kaggle Settings → API**; that account must be able to
   read the existing private model jobs. Paste it into Terminal and press Return.
   No characters appear while you paste. Do not paste tokens into GitHub/chat or
   launcher source. Existing secure local Kaggle credentials can be reused by
   pressing Return without a token. Wait for **“Frozen bundle installed and
   verified”**. No new Kaggle job is submitted.
4. **Double-click `Start_TrustRetina.command`.** First launch installs the exact
   app dependencies in `.venv-mac` (allow several minutes and internet/disk space).
   Subsequent launches reuse them. After the loopback health check passes, your
   browser opens automatically. Keep the Terminal window open while using the
   app. Press **Control-C** to stop it. Starting it again opens the existing
   launcher-managed instance rather than launching another server.

If the bundle is already available privately, copy the entire verified
`inference-v8` folder into `artifacts/` in this checkout and **skip step 3**.
Start verifies all five required files itself. Never obtain weights from an
untrusted public mirror or copy any dataset/prediction folders into the app.

These launch files are on the packaging PR branch until it is merged. A clone:

```sh
git clone --branch phase/08-macos-launcher https://github.com/Saisathwik045/Cv.git
cd Cv
```

Git may prompt to install Apple's command-line tools if unavailable; downloading
and extracting the ZIP avoids needing Git. Once the packaging PR is merged, a
normal `main` clone/ZIP contains the same launch files.

## First-run macOS approval and certificates

Downloaded `.command` files may need explicit approval after you review their
source: try **Control-click → Open**, or use **System Settings → Privacy &
Security → Open Anyway** if macOS offers it. Do not disable Gatekeeper globally.
If the archive lost executable permissions, open Terminal in the project folder
and run:

```sh
chmod +x Start_TrustRetina.command Setup_TrustRetina.command scripts/macos_bootstrap.sh
```

You can also launch with `bash Start_TrustRetina.command` or run setup with
`bash Setup_TrustRetina.command` from that folder. The paths work with spaces.
If you installed Python from python.org and HTTPS certificate verification fails,
run its supplied **Install Certificates.command** in the Python Applications
folder, then retry. Do not disable TLS verification. Use native Terminal/Python,
not Terminal configured to open with Rosetta.

## Artifact provenance and credentials

Setup downloads only:

- `best.pt` from `saisathwikregula/trustretina-full-v5-convnext-tiny-c349c706/1`.
- `temperature.json`, `quality-normalizer.npz`, `candidate-pipeline.json` from
  `saisathwikregula/trustretina-phase6-analysis-ab40b8f2/1`.
- The repository's frozen candidate contract is copied into the bundle.

The existing installer verifies each exact SHA256, including checkpoint
`6871e49772543f3463f9d30dd6c758257022463bf562e21342fd154d7aa0eaf1`
and contract
`a80dce8f27f12645539a0312a3ed96af0b5d10b5d7e1cd522af53b9347409c61`.
Corrupt/partial existing bundles fail explicitly and are not silently overwritten.
No random weights or replacement model are allowed.

A pasted token is passed to the one-time downloader through its child process
environment, never command-line arguments or logs, and is not written by the
launcher. With a token already stored locally, use Kaggle's standard
`~/.kaggle/access_token` or `~/.kaggle/kaggle.json` configuration and restrict the
credential file to your account (`chmod 600`). The setup refuses insecure file
permissions when reusing that configuration. Keep credentials outside the project.
The hidden prompt requires an interactive Terminal. Your account needs permission
for the pinned private jobs; setup does not bypass their access controls.

After installation the app uses no Kaggle SDK, token or ODIR dataset. Kaggle
variables are removed from the server child environment. The app and one-time
recovery environments are separate. Copying the verified bundle is a completely
credential-free setup alternative.

## Lifecycle, diagnostics and troubleshooting

The launcher locates its checkout relative to its own file. A private startup
lock prevents concurrent setup/start races. Reuse requires the saved PID,
checkout/command identity and successful health response. An occupied port from
an unverified/manual server is reported rather than opening an unrelated service
or starting duplicates. Default port is 8501; to choose another locally:

```sh
bash Start_TrustRetina.command --port 8502
```

For a server started by another launcher window, close the original window or
press Control-C there before changing ports. A broken launcher-managed process
is never blindly replaced. A change to the app entry file/dependency lock requires stopping the existing
instance before reuse. The launcher records its own child PID, waits at most
60 seconds for health, and terminates its owned child on startup failure,
Control-C, termination or terminal hangup. No daemon or permanent hosting is set
up. Closing the owning Terminal or stopping/sleeping your computer can interrupt
access; no cloud preview, public tunnel or external hosting is attempted.

Logs are in ignored `.trustretina/launcher.jsonl`, with user-only permissions.
They contain lifecycle stage, exit code, PID, port, Python/platform and lock hash.
Raw pip/Kaggle/server output, environment values, exception messages and uploaded
images are deliberately not persisted: third-party errors can contain secrets or
user data. If setup fails, the displayed stage and exit code identify whether it
was dependency installation, dependency consistency, native runtime imports or
private artifact retrieval;
check internet/certificates, disk space and private-job authorization accordingly.
The app still handles readable invalid-image errors in its interface.

Dependencies are checked against the exact lock before reuse. A changed lock or
missing/wrong version triggers a reinstall; incompatible Python/architecture
requires removing only `.venv-mac` (or `.venv-artifacts-mac`) and retrying. Keep
`artifacts/inference-v8` to preserve the verified model. Do not move a populated
virtual environment between checkouts; recreate it after moving the project.

The existing [app walkthrough](DEMO_V8.md#walkthrough) explains paired-eye uploads,
all eight probabilities, head-only uncertainty, technical quality, Grad-CAM and
research results. This is a research demonstration, not a clinical diagnosis or
automatic acceptance policy. No inference behavior was changed for macOS.

## What was actually verified

This task ran on **Linux**, not a Mac. PyPI metadata checks found compatible
macOS 13 ARM64 wheels, compatible Requires-Python declarations and satisfied
platform-specific dependency constraints for every exact pin in both locks,
for CPython 3.11 and 3.12. This is **not** a native installation/runtime test.
The Mac lock preserves the Linux runtime versions, using the published Mac
Torch/torchvision wheels instead of Linux-only `+cpu` builds.

Portable tests cover version/architecture guards, exact-lock/reuse handling,
concurrent startup prevention, missing bundle failure, private logs, credential
handling, process identity, refusing unrelated occupied ports and health-before-
browser ordering. A real Linux Streamlit fixture verified paths with spaces,
HTTP health, a second launch reusing the same PID, and Control-C cleaning up the
owned server/receipt. The fixture performs no model inference or evaluation.

**Unverified on macOS:** Finder/Gatekeeper approval, native venv/dependency
installation, default-browser opening, native checkpoint loading/latency and
numerical parity. The earlier real CPU checkpoint/parity/Grad-CAM evidence is
Linux evidence and must not be relabeled as Mac evidence. No Mac runner or paid
resource was provisioned. See the committed
[wheel metadata](../results/demo-v8/macos-wheel-metadata.json) and
[portable lifecycle record](../results/demo-v8/macos-portable-verification.json).
