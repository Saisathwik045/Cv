# GPU runner status and implementation

See [Kaggle access checkpoint](KAGGLE_ACCESS_STATUS.md) for current evidence and
resume instructions. The source andrewmvd/ocular-disease-recognition-odir5k is
configured for direct attachment; actual remote contents remain unverified.

## Current result — 29 October 2026 (IST)

The managed proxy works with supported network-enabled command execution.
Previous proxy-repair guidance was based on default-sandbox connection failures
and is superseded. Git fetch and editable data/runner installation succeeded.
Kaggle 2.2.5 and kagglesdk 0.1.38 are installed in
/workspace/.environment/trustretina. Use this virtualenv's bin directory on PATH.

Set KAGGLE_CONFIG_DIR=/workspace/.environment/trustretina-kaggle for client commands
because home is read-only. Keep KAGGLE_API_TOKEN supplied by the environment;
no credential-file copy or manual token argument is needed.

Authentication is blocked by destination policy: installed SDK token introspection
uses api.kaggle.com, but only www.kaggle.com is currently allowed. The client
reported ProxyError; a bounded unauthenticated check confirmed proxy CONNECT
HTTP 403, curl exit 56 and destination HTTP 000. No Kaggle authentication response
was received; token validity and account username remain unknown. Add api.kaggle.com
to the managed allowed HTTP hosts and apply the configuration. Do not rotate the
untested token or request proxy service repair. Preserve proxy/TLS protections.

For exec_command network operations, request sandbox_permissions=
"with_additional_permissions" and additional_permissions.network.enabled=true,
following tool permission requirements. This grants command-level network use
through the existing proxy; managed host policy still applies.

No local smoke receipt was found; remote jobs cannot yet be inspected. No job was
submitted in this continuation, so there is no job ID/version, quota use, CUDA,
remote dataset or retrieved-artifact evidence. Full validation/splits remain pending.
Use the actual authenticated Kaggle owner after access works; candidate GitHub owner
names in older prepared examples are not authenticated account evidence.

## Implemented checks

```sh
# Run network commands with supported network-enabled execution; no token in arguments.
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
trustretina prepare-runner-smoke --owner actual-kaggle-owner --output artifacts/runner/new-gpu-smoke --gpu --dataset-config configs/kaggle-smoke.toml
# Separate, explicit execution after connectivity/account gates pass:
trustretina run-runner-smoke --bundle artifacts/runner/new-gpu-smoke --timeout-seconds 600
```

The execution command submits a private Python script with the configured dataset source and no
runtime internet, requests a 120-second kernel runtime, parses the returned kernel
version and polls that exact version. It retrieves smoke.json/smoke.sha256 from the
same version, verifies checksum plus a per-request nonce and checks real CUDA tensor
arithmetic. Only successful authentication, submission, polling, artifact integrity
and CUDA execution together can set unattended_gpu_execution_verified=true.
Omitting --gpu during preparation provides a CPU-only smoke and can never verify GPU.

A successful GPU smoke **uses GPU quota** even though it does no training. A failed
or timed-out submission may have used quota and is reported as unknown, not zero.
Polling timeout does not cancel a remote job; inspect its pinned version before any
new submission. A durable verification receipt and local lock prevent accidental re-submission
of a bundle. Job ID and returned version are persisted before polling; an existing
version resumes polling/retrieval without another push. Prepared source/metadata are checked
against the fixed nontraining template before submission, preventing the command
from becoming a general training launcher. The same run-runner-smoke command reconnects using its receipt. A submission
interrupted before returning a version stops for inspection of the saved kernel_id,
without automatically resubmitting. Never delete a receipt merely to retry a push.

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
smoke executor). The runner extra targets this API generation; Kaggle 2.2.5 is now
installed locally. Remote authentication and runner compatibility remain unverified
until api.kaggle.com is allowed.
No paid fallback is selected. Kaggle remains a first candidate, subject to account
access, quota and successful smoke evidence. Model-training automation and durable
cross-session checkpoint export remain Phase 2 work.
