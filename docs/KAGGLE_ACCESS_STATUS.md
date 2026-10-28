# Kaggle access checkpoint

## Current evidence — 29 October 2026 (IST)

Repository: Saisathwik045/Cv, TrustRetina. PRs #1–#4 are merged. Latest main used:
7ab736131467f561201eee59ba1322e35bc7a52b. Local checkout was clean before sync.

**Correction to prior checkpoints:** default command-sandbox network restrictions
caused earlier proxy connection failures. The managed proxy is reachable using
supported network-enabled execution. Git fetch and package installation now work.
The earlier conclusion that proxy infrastructure needed repair was unsupported.
An unauthenticated HTTP 401 from www.kaggle.com never tested token validity.

For network commands use exec_command with:

```json
{
  "sandbox_permissions": "with_additional_permissions",
  "additional_permissions": {"network": {"enabled": true}}
}
```

Follow the execution tool's permission requirements. Preserve inherited HTTP(S)
proxies, CA trust and TLS verification; this does not bypass destination policy.

## Installed client and local configuration

/workspace/.environment/trustretina is a Python 3.12 virtualenv with editable
TrustRetina data/runner extras, Kaggle 2.2.5 and kagglesdk 0.1.38 installed.
Use its bin directory on PATH when running trustretina/Kaggle commands.
The read-only home directory prevents the client's default config-directory
creation. Set KAGGLE_CONFIG_DIR=/workspace/.environment/trustretina-kaggle for
client commands. This supported selector points to a writable private directory;
no token was copied there. KAGGLE_API_TOKEN remains supplied by the environment.
Do not print credentials, full config, raw SDK/CLI errors or authorization headers.

## Exact remaining failure

The installed client's production token-introspection endpoint uses
**api.kaggle.com** (observed in installed kagglesdk/kaggle_env.py). Applied managed
policy is current/enforced and lists www.kaggle.com but not api.kaggle.com.
The supported client authentication attempt returned ProxyError. A separate
bounded unauthenticated request to https://api.kaggle.com/ confirmed:

- Proxy CONNECT HTTP 403.
- curl exit 56; destination HTTP 000.
- No Kaggle destination HTTP/authentication response.

This is a host-policy blocker, not a broken proxy or observed token rejection.
Authentication and actual account username remain unverified. No alternate host,
proxy bypass, direct egress, TLS change or credential substitution was attempted.
Do not repeat authentication attempts until the applied host policy changes.

**Minimum action:** add api.kaggle.com to the managed environment HTTP allowlist
and apply it. Preserve www.kaggle.com, restricted mode and the existing token.
Do not rotate an untested credential. No artifact-download hosts are verified yet;
add further hosts only if a real response/policy diagnostic identifies them.

## Receipt inspection and remote acceptance

No verification.json, kernel-metadata.json or smoke.json was found under
/workspace/Cv, /workspace/shared or /workspace/scratch. Prior checkpoints record
no submission. Remote jobs could not be inspected, so account-wide job absence
is not established. No new job was submitted, no remote ID/version was created,
no GPU quota was used, and no training or paid resources were launched.

Structured authentication evidence is in ignored artifacts/runner/authentication.json;
installed dependency versions and local environment evidence are retained beside it.
No CUDA, attached workbook/image, remote polling or checksum-verified remote artifact
evidence exists yet. No full real-data validation or production split preparation
ran, and no dataset version or real-data hashes/manifests have been retrieved.

## Preserved authorization and resume sequence

The user has authorized one short nontraining GPU smoke using available Kaggle
quota, with no paid resources or model training. Do not ask again for that scope.
After applying the missing host rule:

1. Authenticate using the configured token and record the actual Kaggle username
   returned by supported client token introspection; never assume GitHub identity.
2. Inspect saved local receipts and relevant authenticated remote jobs. Resume an
   existing job if present. Never delete a receipt or blindly resubmit after an
   ambiguous response or polling timeout.
3. Otherwise prepare and execute the fixed smoke with the actual account owner:

```sh
trustretina prepare-runner-smoke --owner actual-kaggle-owner --gpu --dataset-config configs/kaggle-smoke.toml --output artifacts/runner/dataset-smoke
trustretina run-runner-smoke --bundle artifacts/runner/dataset-smoke --timeout-seconds 600
```

The source is attached directly:
andrewmvd/ocular-disease-recognition-odir5k. No user download/reupload is required.
The smoke requests a 120-second nontraining runtime and checks real CUDA arithmetic,
data.xlsx structure/hash and original-style decoded eye-image samples. The runner
persists submission intent/kernel ID before push and pinned version before polling,
then checks downloaded artifact checksum and nonce. A timeout may leave a remote job
running; inspect its saved ID. Interrupted submission without a returned version
requires remote inspection, never an automatic new push.

4. Only after smoke success, execute existing full data validation and deterministic
   patient-separated split preparation remotely. Preserve reports/manifests, dataset
   and kernel versions, code revision and input/output hashes outside Git. Fail on
   contradictory/missing labels, duplicates, corrupt inputs or insufficient support;
   do not silently modify the cohort, disable checks or retry seeds.

## Historical implementation acceptance

PRs #1 and #2 merged the foundation and patient validation/split pipeline. PR #3
merged direct dataset attachment and durable resume safeguards with four successful
Python 3.11/3.12 check runs. Previous local evidence was 38 synthetic/mocked tests.
PR #4 recorded earlier access observations; its proxy-repair interpretation is
superseded by this checkpoint. No historical test result is remote GPU/data evidence.
