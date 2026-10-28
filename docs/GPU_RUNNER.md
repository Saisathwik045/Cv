# GPU runner status and implementation

## Observed Phase 1 verdict

**Unattended GPU execution is NOT verified.** No remote smoke job or training was
submitted; no paid resources were provisioned; **no GPU quota was used**. A fixed
nontraining CUDA arithmetic script was prepared in ignored
`artifacts/runner/phase1-gpu-smoke`. Preparing files does not submit or consume quota. The prepared bundle uses the
GitHub login Saisathwik045 as an **unverified candidate owner**; GitHub identity does
not establish Kaggle identity. Regenerate a fresh bundle with the actual Kaggle
username after authentication, before submitting.

Actual `check-runner` observations in this session:

- Kaggle CLI/package is not installed (client_version null).
- KAGGLE_API_TOKEN, KAGGLE_USERNAME/KAGGLE_KEY, token-file and legacy-file presence
  checks are false. No credential values or secret-file contents were printed.
- Managed readiness is current and reports no bound secrets/runtime variables or
  outbound identity. Presence alone would not establish valid authentication.
- `/etc/codex/network-policy.json` and managed readiness both show an enforced
  restricted allowlist without www.kaggle.com. No VPN/TCP grant is configured.
- Requests to Kaggle, PyPI and GitHub through inherited HTTP(S)_PROXY fail connecting
  to the managed proxy at `proxy:8080` (curl exit 7). This is a proxy transport
  failure before receiving a Kaggle HTTP/auth response. Policy also needs updating
  even after proxy transport recovers. The GitHub connector is a separate working
  path; it cannot be used to proxy Kaggle requests.

## Account/environment actions that require your access

1. In the attached Codex cloud environment's configuration, keep restricted network
   mode and add **`www.kaggle.com`** to allowed HTTP hosts. Do not disable security
   controls, change credentials in code, unset HTTP_PROXY/HTTPS_PROXY, disable TLS
   verification or attempt direct egress. The installed client may later expose
   additional API/artifact host requirements through policy diagnostics; only add
   those exact observed hosts. No artifact-download hostname is verified yet.
2. Restart/recreate the managed environment using that configuration, or have its
   environment/platform administrator repair the unavailable proxy service at
   **`proxy:8080`**. Package-manager/GitHub hosts are already allowed; allowing
   Kaggle alone does not fix a proxy service connection failure. No project-code
   change can grant egress or repair the managed sidecar.
3. Generate an API token in [Kaggle API settings](https://www.kaggle.com/settings/api)
   and use the managed environment configuration's **secret binding** to expose it
   as environment variable **`KAGGLE_API_TOKEN`**. Do not paste it into chat or commit
   it. There is no installed Kaggle client here, so no installed-client credential
   contract has been empirically tested. The audited official 2.2.5 source/docs
   support KAGGLE_API_TOKEN and `~/.kaggle/access_token`; legacy alternatives are
   **KAGGLE_USERNAME + KAGGLE_KEY** or `~/.kaggle/kaggle.json` selected with
   KAGGLE_CONFIG_DIR. One token secret is the minimal recommended setup; do not
   configure every alternative. No Kaggle-internal secret is needed by this smoke.
4. Check the account's verification/GPU eligibility prompts, available accelerators,
   remaining quota and session/output limits in Kaggle. No quota or accelerator
   availability is assumed. Make the original ODIR dataset, including data.xlsx,
   accessible privately and accept source terms if necessary.

GitHub main protection remains a separate recommended account setting; no additional
GitHub credentials are needed for this session's working connector. PR #1 was
reviewed and merged; Phase 1's PR must remain unmerged for review.

## Implemented checks

```sh
# After package/network access works; agent can install this, no token in arguments.
python -m pip install -e ".[runner]"
trustretina check-runner
```

check-runner performs a bounded read-only HTTPS probe and, when a CLI plus supported
credential presence is observed, runs `kaggle kernels list --mine`. It exits nonzero
when authentication cannot be verified. It always leaves submission, polling,
retrieval and GPU verification false because no job runs during preflight.
Raw CLI stdout/stderr are captured, never printed or saved; failures report bounded,
structured reasons without tokens. Missing dependencies and failures are not success.

```sh
# Preparation only: replace the owner below with the authenticated Kaggle username.
trustretina prepare-runner-smoke --owner Saisathwik045 --output artifacts/runner/new-gpu-smoke --gpu
# Separate, explicit execution after connectivity/account gates pass:
trustretina run-runner-smoke --bundle artifacts/runner/new-gpu-smoke --timeout-seconds 600
```

The execution command submits a private Python script with no dataset sources or
runtime internet, requests a 120-second kernel runtime, parses the returned kernel
version and polls that exact version. It retrieves smoke.json/smoke.sha256 from the
same version, verifies checksum plus a per-request nonce and checks real CUDA tensor
arithmetic. Only successful authentication, submission, polling, artifact integrity
and CUDA execution together can set unattended_gpu_execution_verified=true.
Omitting --gpu during preparation provides a CPU-only smoke and can never verify GPU.

A successful GPU smoke **uses GPU quota** even though it does no training. A failed
or timed-out submission may have used quota and is reported as unknown, not zero.
Polling timeout does not cancel a remote job; inspect its pinned version before any
new submission. An exclusive verification receipt prevents accidental re-submission
of a bundle, including after ambiguous failures. Prepared source/metadata are checked
against the fixed nontraining template before submission, preventing the command
from becoming a general training launcher. A future receipt-based reconnect command
is not yet implemented; use the pinned upstream status/output commands after an
interrupted verification. Never delete a receipt merely to retry a push.

Tests use mocked CLI responses and synthetic downloaded artifacts; those tests are
not evidence that any Kaggle operation succeeded remotely. Remote behavior, artifact
download hosts and cross-session checkpoint durability remain unverified.

## Sources and limits

Read using the authenticated GitHub connector, which resolves former Kaggle/kaggle-api
to Kaggle/kaggle-cli:

- [Authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)
- [Kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [Metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
- [Audited API implementation](https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/api/kaggle_api_extended.py)

The audited upstream package identifies itself as 2.2.5. Its CLI documents script
push/status/output; push runs by default (no --no-run is passed by the explicit
smoke executor). The runner extra targets this API generation; actual package
installation and version compatibility remain to be checked after network recovery.
No paid fallback is selected. Kaggle remains a first candidate, subject to account
access, quota and successful smoke evidence. Model-training automation and durable
cross-session checkpoint export remain Phase 2 work.
