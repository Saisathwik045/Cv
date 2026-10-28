# Kaggle access checkpoint

## Latest recheck — 28 October 2026, 22:58 IST

This section supersedes the older access observations below; those observations
are retained as history.

- Verified project: Saisathwik045/Cv is TrustRetina. PRs #1–#3 are merged.
  Latest remote main: cae2fbc915e52c52d571fbbe8e8551066475f39f.
  PR #3 head c59be415ea1237fa087437db2201bd27e65f6855 has four successful
  Python 3.11/3.12 check runs. Its submitted-review list is empty.
- Managed runtime is running with current observations. Restricted HTTP policy is
  enforced and includes www.kaggle.com; package/GitHub hosts are also allowed.
  KAGGLE_API_TOKEN readiness is ready and its process presence is true.
  No token value or secret-file content was read, logged or printed.
- Actual access: a single bounded Kaggle request failed with curl exit 7,
  unable to connect to proxy:8080; HTTP status 000. Git fetch failed at the
  same proxy. This occurs before Kaggle authentication or any destination HTTP
  response; it is not an observed token rejection or host-policy denial.
- Client installation: one attempt for kaggle>=2.2.5,<3 with --retries 0
  --timeout 10 returned no package candidate. No client was installed; package
  availability on PyPI is not established by this failure. No retries followed.
- Authentication and the actual Kaggle username could not be established.
  GitHub identity was not substituted.
- Receipt inspection found no verification.json, kernel-metadata.json or
  smoke.json under /workspace/Cv, /workspace/shared or /workspace/scratch.
  Remote job inspection is blocked, so this is not proof the account has no jobs.
  No new submission, remote ID, polling, CUDA evidence, dataset attachment,
  artifact retrieval, real-data validation or split publication occurred.
  No quota was consumed by this session; no training or paid resources launched.
- Local checkout remains the initial commit because shell fetch failed.
  The documentation update is published through the working GitHub connector,
  based on current remote main; completed phases are not reconstructed.

**Minimum next action:** recover the managed proxy using a fresh environment or
the Codex support flow. The user has already supplied the host rule and token
variable; do not ask them to add those again or rotate an untested token.
No exposed project-side setting can repair the managed proxy service. Do not
bypass it, unset proxy variables or disable TLS.

After an actual environment change, install/authenticate the supported client and
determine its authenticated username. Inspect remote jobs before any submission,
in addition to local receipts. Resume an existing job when present; otherwise
the single short nontraining GPU smoke remains authorized. Attach the configured
dataset directly. Only after successful CUDA/dataset/status/artifact verification,
run existing full validation and patient-separated split preparation remotely.
Retain dataset/kernel versions, source revision, reports/manifests and hashes
outside Git. Any label, duplicate or support failure must remain a failure.
No new permission for that already-authorized sequence is required.

## PR #2 review

Reviewed exact head 6a361f2910efc8768ceae7bbde456c5539583ef1 against Phase 1
code acceptance: authoritative XLSX parsing, paired patient labels, input validation,
deterministic grouped splits, support/leakage checks, immutable manifests and honest
synthetic-only evidence. All four latest GitHub check runs (push/PR, Python 3.11/3.12)
were successful. All 34 local tests also passed. No code acceptance blocker was
found. PR #2 was merged with expected-head protection; merged main is
16f0a5c64f991a73f383bb78d8fdfe9f1740a9d1. Real-data acceptance is still pending.

## Dataset configuration

Use the existing source directly:
`andrewmvd/ocular-disease-recognition-odir5k`.
`configs/kaggle-smoke.toml` sets this reference and the remote input mount location.
Smoke metadata includes it in dataset_sources. No download/reupload by the user is
required. Content is **unverified** until Kaggle actually attaches and executes it.
The script discovers data.xlsx, verifies XLSX structure, hashes it, finds the expected
Training Images directory and decodes original-style eye-image samples. This checks
layout/content presence; full workbook/image validation and source authenticity are
separate. A repackaged CSV is never substituted for data.xlsx.

## Single diagnosis in this continuation

- Current managed readiness: enforced restricted HTTP policy, no Kaggle host entry,
  no secrets/runtime-variable bindings and no outbound identity.
- Presence-only checks: KAGGLE_API_TOKEN absent; KAGGLE_USERNAME and KAGGLE_KEY absent;
  token file and legacy kaggle.json absent. No values/files were read or printed.
- One Kaggle HTTPS probe failed with curl exit 7: could not connect to proxy:8080.
  This is **proxy connection failure before a destination-policy/auth response**,
  not observed evidence of an HTTP allowlist rejection. Independently, the policy
  snapshot excludes www.kaggle.com, so that setting also needs to change.
- One bounded install attempt (`--retries 0 --timeout 10`) for kaggle>=2.2.5,<3
  failed: no package candidate was obtained. No Kaggle client is installed. This
  package-resolution failure does not prove that version is unavailable on PyPI.
  Installed-client authentication behavior therefore cannot be verified here.
- Network retries stopped after this diagnosis. No direct egress, proxy bypass,
  security-control change or credential disclosure occurred.

## Supported configuration actions

The available tool interface exposes readiness and policy but **no writable
configuration or view of the user's settings screen**. Supported configuration
fields are restricted HTTP host rules and secret-variable bindings. In the attached
cloud environment configuration editor, if those controls are exposed:

1. Keep restricted networking. Add `www.kaggle.com` to allowed HTTP hosts. Existing
   PyPI/package hosts are already allowed. Add artifact hosts only if a later real
   response/policy diagnostic identifies them; none is verified yet.
2. Add a secret binding named `KAGGLE_API_TOKEN`, using a token generated at
   https://www.kaggle.com/settings/api. Store the value only in the secret field,
   not a plain-text variable, source file or chat. Audited upstream 2.2.5 docs/source
   support this token name; legacy alternatives are KAGGLE_USERNAME + KAGGLE_KEY or
   kaggle.json. A token binding alone is the recommended minimum.
3. Save/apply and select a fresh cloud session so the configuration is applied.
   If only a general Internet access toggle is visible, leave the managed proxy and
   security controls intact; that toggle is not evidence of host/secret configuration.
   If no configuration editor is available, use the product's environment-access/
   support flow rather than attempting changes inside the container.

There is no user-facing setting here to repair proxy infrastructure. A fresh session
may recover a failed runtime; if proxy connectivity still fails, report the cloud
session failure through Codex support. The user is **not** asked to repair proxy
infrastructure. Exact click-path labels cannot be verified from these read-only
tools; do not invent UI navigation.

## Execution authorization and resume point

The user has authorized **one short nontraining GPU smoke using available Kaggle
quota**, no paid resources and no model training. This authorization persists; do
not ask again merely because credentials/network become available. Confirm the
actual authenticated Kaggle owner before preparing/submitting the final bundle;
GitHub identity is not Kaggle identity. Prepared bundles with candidate owner names
must not be mistaken for authenticated jobs.

Preparation example (replace the candidate owner with the authenticated Kaggle owner):

```sh
trustretina prepare-runner-smoke --owner actual-kaggle-owner --gpu --dataset-config configs/kaggle-smoke.toml --output artifacts/runner/dataset-smoke
```

The separate run-runner-smoke command submits only after access works. It writes
kernel_id and submission intent before push, fsyncs the returned pinned kernel_version
before polling, and writes a verification receipt. Rerunning with the same bundle
resumes polling/retrieval and never pushes an already submitted job again. Ambiguous
submission without a returned version stops with the saved kernel_id for remote
inspection; it never blindly resubmits. A local lock prevents concurrent submission.
After an unclean process kill, verify its recorded PID is no longer running before
removing only a stale verification.lock. Keep verification.json intact.

The configured smoke requires dataset evidence as well as CUDA arithmetic and a
nonce/checksum-verified artifact before marking GPU execution verified. All polling
is bounded. The 120-second requested runtime performs no training. If a job times
out locally it may remain queued/running remotely; inspect its saved ID instead of
submitting a replacement. CPU smoke can never certify GPU access.

**Actual operations so far:** no authentication success, no job submission, no
remote job ID, no polling, no CUDA evidence, no dataset attachment evidence and no
artifact retrieval. No GPU quota was used. After access is restored, perform the
single authorized smoke. Only if it succeeds, run existing data validation/split
preparation remotely, retrieve its reports/manifests and record the exact source
revision, dataset/kernel version and input/output hashes. Source-data problems must
still fail closed; do not silently fix labels, disable duplicate checks or train.
