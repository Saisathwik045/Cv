# Direct private Kaggle recovery and launch on a Mac

This route avoids exporting the cloud archive or uploading the checkpoint to
another service. Your Mac downloads only four original pinned Kaggle objects,
verifies their SHA256 identities and assembles the five-file inference-v8 bundle.
It does not train, recalibrate, submit jobs or run model inference during recovery.
The existing dashboard starts afterward; predictions require your Analyze action.

## Minimum local steps

Use **Apple silicon, macOS 13+, native ARM64 Python 3.12.x** (3.11.x also works).
Intel/Rosetta and Python 3.13+ are not supported by this locked launcher. Native
Mac runtime/parity is still unverified; Linux and portable tests are not Mac evidence.

If native Homebrew is already installed:

```sh
brew install python@3.12
mkdir -p "$HOME/TrustRetina"
git clone --branch phase/17-bounded-mac-recovery https://github.com/Saisathwik045/Cv.git "$HOME/TrustRetina/Cv"
cd "$HOME/TrustRetina/Cv"
bash Setup_and_Start_TrustRetina.command
```

The branch above is the implementation/review branch; after it is merged, use
current main instead. Preserve any existing checkout rather than cloning over it.
Without Homebrew, install a compatible signed Python 3.12 universal2 installer
from python.org. GitHub Code → Download ZIP for this branch is an alternative to
Git; extract the complete folder and run the same command from it.

The command automatically creates separate recovery/app environments and installs
their existing exact locks. When prompted, paste your **Kaggle API token into the
hidden Terminal prompt**. Input is not displayed, put in command arguments, logged
or saved by the launcher. Never paste a token into chat or a source file.
The activated account must be `saisathwikregula` and retain access to both original
private versions. Press Return without a token only to use already configured,
owner-only Kaggle credentials on your Mac.

Allow internet access and disk space for both Python environments in addition to
the downloader's minimum **2 GiB free** gate. First setup can take several minutes.
After verified recovery, the original launcher opens **http://127.0.0.1:8501**.
Keep Terminal open; press **Control-C** to stop. A local address only works on the
Mac running the application. No cloud hosting, tunnel or public endpoint is created.

For subsequent runs, double-click **Setup_and_Start_TrustRetina.command**, or use
the same bash command. A complete verified bundle is reused before prompting for
credentials or installing recovery dependencies. Only the first actual recovery
needs Kaggle access. Finder may require Control-click → Open after source review;
do not disable Gatekeeper or TLS verification.

## Exact recovery contract

| Object | Version | Exact bytes |
|---|---|---:|
| full/best.pt | saisathwikregula/trustretina-full-v5-convnext-tiny-c349c706/1 | 334,159,877 |
| full/temperature.json | saisathwikregula/trustretina-phase6-analysis-ab40b8f2/1 | 1,020 |
| full/quality-normalizer.npz | Same Phase6 version above | 79,547 |
| full/candidate-pipeline.json | Same Phase6 version above | 1,550 |

Expected private source size: **334,241,994 bytes**. The fifth file is the committed
candidate-contract.json, SHA256-checked locally. All expected hashes are loaded
from the committed frozen contract and cross-checked against
[the original recovery record](../results/demo-v8/artifact-recovery.json).
The numerical limits are not inferred from unreliable listing-size fields.

- HTTPS file host: **www.kaggleusercontent.com** only. API metadata/authentication
  uses Kaggle's SDK endpoints separately. Signed URLs remain in memory.
- Version-label `v1`, exact file-name matching, unique matches and bounded metadata pagination.
- No `kernels_output`, kernel-log writes, datasets, archive-wide output downloads or jobs.
- Automatic redirects disabled; **all redirects are rejected** without reading their bodies.
- Require HTTP200, exact Content-Length, identity encoding and no transfer-encoding
  ambiguity before consuming a file body. Unexpected responses fail closed.
- Bounded streams, exact per-file sizes and full SHA256 verification before assembly.
- Maximum cumulative transfer charge **668,483,988 bytes**, including interrupted
  reads and retained partials; a read is reserved in persistent state before it starts.
- No automatic retries. Up to two requests per object across local invocations,
  within the cumulative ceiling. A network interruption may permit one manual
  rerun; a security/integrity mismatch blocks further recovery pending review.
- Good recovered files are rehashed and reused; corrupt existing artifacts are
  preserved and never overwritten. Final assembly is verified and promoted only
  after all five hashes pass. No Torch import or checkpoint deserialization occurs.

## Private storage and provenance

The combined command sets:

```text
TRUSTRETINA_BUNDLE=$HOME/Library/Application Support/TrustRetina/inference-v8
```

All paths are quoted by the wrapper. The parent TrustRetina folder, recovery
workspace and bundle use 0700; private files use 0600. Symlink paths and insecure
existing private directories are rejected rather than silently changing them.
An adjacent `.inference-v8-recovery` directory retains the approval reference to
the explicit local setup invocation, plan/tool/dependency identities, byte ledger, partials and completion
receipt. Treat these as private. Do not put them or the bundle on GitHub.
Application runtime uses no Kaggle credentials; the server child removes Kaggle
variables. Downloading again on the Mac is not a transfer of the prepared cloud
archive. Neither location is a verified durable encrypted backup.

For a different user-owned private location, set it explicitly:

```sh
TRUSTRETINA_BUNDLE="$HOME/TrustRetina/private/inference-v8" \
  bash Setup_and_Start_TrustRetina.command
```

Use that same setting for future launches. Finder does not inherit a prior
Terminal export. The combined command defaults to its stable private location;
the old standalone Start command retains its older default. To launch without
setup through that old command, explicitly supply the bundle path:

```sh
TRUSTRETINA_BUNDLE="$HOME/Library/Application Support/TrustRetina/inference-v8" \
  bash Start_TrustRetina.command
```

## Inspection and failure handling

Public plan inspection requires no SDK authentication or model bytes:

```sh
python3.12 scripts/recover_inference_bundle.py --plan
```

For controlled recovery diagnostics after launcher setup, the isolated recovery
environment can invoke the same bounded command directly:

```sh
.venv-artifacts-mac/bin/python -B scripts/recover_inference_bundle.py \
  --allow-recovery --destination "$HOME/Library/Application Support/TrustRetina/inference-v8"
```

Do not delete the ledger to bypass a limit. Preserve failed files/state for review.
No more than the two permitted attempts are allowed; integrity mismatches must
not be solved by accepting a different hash, version, model or larger file.
If an existing destination fails verification, preserve it and investigate.
The cloud model archive remains unchanged and can be a separately approved
alternative, not a fallback download invoked by this script.

After setup, follow the synthetic live demonstration checklist in
[the demo walkthrough](DEMO_V8.md) and [Mac guide](MACOS_QUICKSTART.md).
The command prepares [generated non-patient PNGs](../examples/synthetic-demo/README.md)
in `~/Library/Application Support/TrustRetina/demo-images`: upload `left.png` and
`right.png`; use `replacement.png` to test upload replacement. Image files are
not committed. No cloud archive or patient-image transfer is needed.
Predictions on generated images are operational checks, not medical evidence.
The original epoch-2/224 px checkpoint, temperature 0.7226021421263825,
p>=0.5 rule and REVIEW REQUIRED behavior remain frozen; V14/V15 are not substituted.

## Verification boundary

Synthetic tests exercise exact selection, body rejection before reads, hashing,
private permissions, symlinks, byte reservations, interrupted reuse and retry
ceilings. Launcher fixtures verify hidden-token handling and setup-before-start
ordering. These checks activate no real Kaggle account and download no model.
Only your local execution can establish Mac installation, account entitlement,
actual recovery, browser opening, performance and numerical parity.
