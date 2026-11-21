# Private Streamlit Community Cloud deployment

This is a separate deployment entry point for a stable browser URL, not a change
to the original frozen scientific pipeline. Local `127.0.0.1` points to whichever
machine opens it; the cloud workspace's loopback address cannot be used directly
from the student's Mac. A hosted URL must be obtained from the hosting service.
No app URL is claimed until the actual deployment succeeds.

## Account and source preparation

The user selected Streamlit Community Cloud on 21 November 2026. Its official
documentation describes free hosting and one private app per account. Apps inherit
repository visibility by default. Deploy from an **owner-controlled private source
repository**, since `Saisathwik045/Cv` is public. Do not change the original repository's
visibility or upload any weights there.

The connected GitHub integration rejected `createRepository` with `Resource not
accessible by integration`. No private repository was created. The owner must
create/import the private source through their own account, or grant appropriate
access through normal GitHub controls. Do not paste tokens in chat. The current
code is prepared in the phase PR; it is not silently merged into main.

After reviewing/merging the deployment PR, use GitHub's
[repository import](https://github.com/new/import) with source
`https://github.com/Saisathwik045/Cv.git`, owner `Saisathwik045`, name
`trustretina-streamlit-private`, and **Private** visibility. This copies public
source/history only. Never include private workspace directories or artifacts.
Verify that the imported main contains `deploy/community_cloud/app.py` before
deployment. Repository access/OAuth prompts must be reviewed by the owner; private
repository access can require additional GitHub permissions.

## Exact Create app settings

From the owner's Streamlit dashboard select **Create app → Deploy from GitHub**:

| Field | Value |
|---|---|
| Repository | `Saisathwik045/trustretina-streamlit-private` — only after the owner creates it |
| Branch | `main` containing the reviewed deployment code |
| Main file path | `deploy/community_cloud/app.py` |
| Python | `3.12` in Advanced settings |
| App URL | Choose an available name; use the actual URL returned by Streamlit |

The entry-point-local requirements file reuses the exact Linux CPU app lock.
A root requirements.txt also references that same lock if the root dashboard is
selected accidentally; it does not configure private model storage. Both install
paths require binary wheels, so an unsupported interpreter fails instead of
attempting to build NumPy/PyTorch from source. Select Python 3.12 explicitly.
The entry-point-local Streamlit configuration binds the **hosted backend** to
`0.0.0.0` with CORS/XSRF enabled and telemetry disabled. The platform's private
authentication gateway must protect it. The original local loopback configuration
remains unchanged. Do not use a tunnel or expose the backend directly.

## Confirm private access before private setup

1. Before adding any model credential, verify the actual app's **Settings → Sharing →
   Only specific people can view this app** and keep access to the owner initially.
2. Test its actual URL in a signed-out/incognito browser. It must require sign-in/
   deny app content to unauthorized viewers. A local Boolean is not authentication.
3. In **App settings → Secrets**, configure:

```toml
[trustretina]
owner_only_confirmed = true
kaggle_api_token = "ENTER_YOUR_OWN_AUTHORIZED_TOKEN_IN_STREAMLIT_ONLY"
```

The example contains no real token. Use the owner's Kaggle account already
authorized for the original pinned sources. Do not send a token to chat, GitHub,
query strings or browser form uploads. Streamlit retains this secret on its
platform; only enable it after approving that private destination. The app code
does not copy the token to files or pass it in subprocess arguments. It sends it
via stdin to an isolated recovery process. The platform secret remains configured
until its owner removes it; this is not a claim that credentials disappear from
Streamlit's secret store after recovery.

The initial setup screen loads no model and performs no download. The
`owner_only_confirmed` setting is an operator attestation, not independent proof
of platform access control. The owner must actually check the platform setting.

## Explicit bounded original-bundle recovery

Select **Prepare verified original bundle** once. This creates an isolated SDK
venv with the existing recovery dependency lock, keeping its protobuf/other pins
out of the app environment. No recovery credential is forwarded to package
installation. Recovery receives its own token only after dependency preparation.

The child reads only these four exact objects:

| Pinned source | Object | Expected bytes |
|---|---|---:|
| `saisathwikregula/trustretina-full-v5-convnext-tiny-c349c706/1` | `full/best.pt` | 334,159,877 |
| `saisathwikregula/trustretina-phase6-analysis-ab40b8f2/1` | `full/temperature.json` | 1,020 |
| same pinned analysis version | `full/quality-normalizer.npz` | 79,547 |
| same pinned analysis version | `full/candidate-pipeline.json` | 1,550 |

Total exact private payload: **334,241,994 bytes**. The public committed
`candidate-contract.json` is the fifth bundle file. Expected SHA256s come from
the hash-verified original public contract, never from user-entered hashes or a
different candidate. The broad `kernels_output` installer is not called.

Controls: authenticated metadata requests use exact `v1` labels and file names;
metadata pagination is capped. Only HTTPS `www.kaggleusercontent.com` is allowed;
redirects, unexpected statuses/sizes/encodings and chunked responses are rejected
before reading a private body. Streams consume at most the exact expected bytes,
then require the frozen SHA256. At least 2 GiB free is required. Files are 0600,
private data directories 0700, with symlinks rejected and no verified bundle overwrite.

The 668,483,988-byte cumulative budget is enforced by durable **full-file reservations
before each private request**, under an exclusive process lock. Failed/short requests
therefore consume conservative charged budget, not necessarily their exact received
bytes. There is no automatic retry. Failed staging is retained privately and never
published as a bundle. An interrupted ledger update fails closed. Only after all
five hashes pass is the staged directory published and configured via
`TRUSTRETINA_BUNDLE`. Hash-bound owner-action records and receipts remain private.

SDK preparation has a 300-second timeout; recovery has a 600-second timeout.
Rebuilds may replace ephemeral host storage, including the local budget ledger;
this is a per-storage-lifetime budget, not a cross-account lifetime limit or durable
backup. Missing assets after a rebuild require a new explicit owner setup action,
never automatic repeated downloads. No patients, historical-final data, kernel logs,
training artifacts, V14 candidates, GPU jobs or fitting are involved.

## Verify actual hosted operation

After successful setup the wrapper verifies the bundle and opens the unchanged
dashboard. It hides CUDA and uses the original epoch-2 ConvNeXt-Tiny, 224 px,
T=0.7226021421263825 and p>=0.5. Every output remains REVIEW REQUIRED.

Check the real hosted URL, owner-only access and signed-out denial, all pages,
readiness, then one approved generated-input CPU demonstration: eight outputs,
paired Grad-CAM, JSON export, replacement/reset. Observe memory/latency on the
actual host; a local test cannot certify host capacity. The readiness page allows
the explicitly hosted backend binding but still reports platform access control,
external viewing and uptime as unverified. Never infer authentication from health.

The stable URL replaces the unusable Mac loopback address. It is not an uptime
guarantee, a durable private backup or clinical validation. If the host exceeds
its resource allowance or blocks authenticated source access, stop and report
that error; do not substitute models, enable paid hardware, turn the app public
or weaken access protections automatically.

## Evidence and remaining handoff

Tests use fake byte streams/metadata and mocked UI secrets, never real credentials
or private downloads. They cover exact scope, private permissions, host/redirect/
size refusal before body read, durable budget charging, short/hash failures,
publication only after complete verification, isolated token handling and a gated
setup page. The full CPU suite remains required. Hosting itself is not verified
until the owner creates the private repository/app and the actual hosted checks pass.

Primary documentation checked 21 November 2026:
[Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud),
[private sharing](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app),
[deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app),
[account setup](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/create-your-account),
[permissions/configuration limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/status).


## Troubleshooting observed hosted failures

A rendered upload form does not establish that PyTorch or the model is available.
Observed owner screenshots showed Python 3.14 and direct streamlit_app.py startup
with no torch. A later screenshot showed source builds and dependency-install
failure; its exact underlying installer error was not visible. The root CPU
requirements reference fixes dependency discovery for an accidentally selected
root entry point. Python must still be selected as 3.12 on each new app.

Deploy **deploy/community_cloud/app.py**, not streamlit_app.py, from
**Saisathwik045/trustretina-streamlit-private**. That private source mirror was
created from the public original main; no private artifacts belong in Git.
The wrapper now checks supported Python and CPU imports before any model setup.
Its runtime PASS means imports only, not model verification or clinical validity.
The manual predictor, guided demo and inspector handle missing CPU imports without
printing raw import exceptions. Readiness distinguishes an absent bundle from
one that cannot pass identity verification, without printing private paths.

Do not respond to a missing bundle by using the broad historical installer,
changing model candidates or silently redownloading. Verify owner-only sharing
and use the existing explicit bounded setup. Local model verification cannot
certify the hosted install, private gateway, WebSocket route or available memory.
Codespaces remains an optional alternative, not a deployed service: billing/quota
access was denied, so no potentially paid instance was started.
