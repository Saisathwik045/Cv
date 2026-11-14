# V15 selective artifact recovery infrastructure

This phase prepares recovery software and a separate Python environment after
merged PR #29/main `e29c4641d17ffb5cafd9e1c8af5c29abb20744f6`. Authorization
covers installation, public connectivity, synthetic tests and one PR only.
**No private recovery, credential access, real fitting, inference or job submission
is authorized or performed.** Historical results, frozen assets, thresholds,
normalizers, demo and V14/V15 scientific protocols remain unchanged.

## Separate environment, same cloud machine

The original `/workspace/Cv/.venv` remains the CPU analysis/app environment.
Recovery uses `/workspace/.venvs/trustretina-recovery`, Python 3.12.14, installed
from the unchanged `requirements/artifact-recovery.lock` (Kaggle 2.2.5,
kagglesdk 0.1.38). No second cloud environment is created. The recovery module
uses the standard library plus the installed SDK/requests; it does not require
NumPy, Torch or the analysis environment's optional packages.

Reproducible setup in this existing environment, if the recovery virtualenv is
missing on a later task:

```sh
.venv/bin/python -m venv /workspace/.venvs/trustretina-recovery
/workspace/.venvs/trustretina-recovery/bin/python -m pip install -r requirements/artifact-recovery.lock
/workspace/.venvs/trustretina-recovery/bin/python -m pip check
/workspace/.venvs/trustretina-recovery/bin/python -B scripts/check_recovery_environment.py
```

The check validates every lock pin, real SDK imports, request/version-label
compatibility and CLI help in a child with empty HOME/config and no inherited
credential variables. It replaces authentication **before** Kaggle's package
initializer can call it and disables socket connections. This verifies software,
not account access. Do not run ordinary `import kaggle`, `kaggle --help`,
`check-runner` or authentication probes in the normal environment under this
preparation authorization: Kaggle's initializer attempts credential discovery.

Public unauthenticated probes observed Kaggle website HTTP200, kernels list401,
API root404, PyPI200 and GitHub200. These are service responses, not evidence of
private-version accessibility, signed-download routing or account entitlement.
No credential locations/values were inspected. The environment remains network
restricted; additional signed-download hosts, if encountered later, need explicit
review rather than a wildcard grant.

## Source inspection and recovery scope

The requested `scripts/trustretina_v15.py` is not an existing file. V15 is the
`scripts/trustretina_v15/` package and `scripts/calibration_reliability_v15.py` CLI.
The new infrastructure adds `scripts/recover_v15_artifacts.py` and the separate
`trustretina_v15.recovery` module. No old source file is edited, preserving V14's
full-src/source/configuration/protocol hashes.

Existing `candidate_recover.py` recovers training tensors/ImageNet weights and
role caches, including inputs unnecessary for saved-prediction V15 analysis.
`candidate_experiment.py retrieve` writes private state and exports a public
frozen-plan snapshot; its high-level transport can retrieve kernel logs. Neither
is used by the new recovery workflow. No existing controller is invoked here.

Only these two immutable original output versions are supported:

- `saisathwikregula/trustretina-v14-fit-320-7bbe5ade/1`
- `saisathwikregula/trustretina-v14-fit-224-779a38cd/1`

The public registry and frozen plan are verified against the V15 pinned SHA256s
before creating an inventory. Every remote object is an exact `full/<filename>`
from those records. No raw images, original datasets, training `.npy` caches,
ImageNet weights, historical-final files, arbitrary files, sources or logs are
requested by the downloader.

Two profiles distinguish numeric inputs from current validator dependencies:

| Profile | Exact count | Checkpoint binaries | Intended use |
|---|---:|---|---|
| `metadata` (default) |32|Excluded|Saved validation/calibration predictions, MC passes, quality, reports and integrity metadata; not a complete fitting bundle|
| `analysis` |36|Four opaque best/last files|All 17 registry files plus stage-report.json per run, as required by the unchanged V15 loader|

The six prediction/MC NPZ files per model and quality-features.npz are the numeric
inputs. `selected-checkpoint.json`, `experiment-spec.json`, `stage-report.json`
and dependency/progress/history/metrics/checksum files establish provenance or
remain required by the existing loader. Both profiles include checksum sidecars.
The `analysis` profile includes best.pt/last.pt solely for byte hashing; no model
loading or inference occurs. These four files are not mathematically needed to
fit temperatures, but omitting them cannot satisfy the current exact loader.
They therefore need an explicit checkpoint-byte permission in the later approval.
No frozen artifact or loader guard is relaxed to avoid that requirement.

## Safe public planning command

```sh
/workspace/.venvs/trustretina-recovery/bin/python -B scripts/recover_v15_artifacts.py plan --profile metadata
/workspace/.venvs/trustretina-recovery/bin/python -B scripts/recover_v15_artifacts.py plan --profile analysis
```

These commands read only public repository metadata, do not import Kaggle, create
no private storage and require no credentials. Output is a public plan envelope
with `plan_sha256` and `plan`; it has only registered file/version/hash identities,
not patient records. An optional shell redirect can save it outside the repository
for review. No disabled/real approval file is generated automatically.

## Separate recovery authorization — not granted in this phase

A future user must explicitly authorize private recovery, approved storage and
credential activation. Store the resulting approval privately. Its required keys:

- `approved`: true only after an actual separate user authorization.
- `scope`: `v15_selective_artifact_recovery`.
- `plan_sha256`: canonical SHA256 from the reviewed exact public plan.
- `private_root`: the exact absolute approved output directory outside the repo.
- `user_approval_reference`: genuine reference to that user authorization.
- `credential_access_authorized`: true, explicitly permitting activation of the
  approved Kaggle identity for these versions; this is **not** enabled here.
- `allow_checkpoint_bytes`: false for metadata, true only for analysis.
- `calibration_fitting_authorized`: false. Fitting authorization is never created
  or accepted by this recovery tool.
- `maximum_file_bytes`, `maximum_transfer_bytes`: positive integer storage/transfer
  budgets chosen after review. Do not invent expected artifact sizes. Retained
  partial/final bytes conservatively count against the latter across retries.
- `download_hosts`: exact reviewed HTTPS hostnames for signed downloads; no
  wildcards, HTTP URLs or unreviewed redirect destinations.

Approval fields are operator-enforced records, not cryptographic signatures.
Never fabricate an approval reference. Recovery and V15 fitting approvals use
different scopes and exact bindings; neither authorizes the other. Actual Kaggle
account ownership/access and current retention of the two private versions remain
unverified. The operator must configure credentials through an approved secure
mechanism later, without sending values to chat, Git or shell logs. No credential
values are accepted as CLI parameters.

After separate approval only, the working recovery command is:

```sh
/workspace/.venvs/trustretina-recovery/bin/python -B scripts/recover_v15_artifacts.py recover \
  --profile metadata --plan-file /approved/private/reviewed-plan.json \
  --approval-file /approved/private/recovery-approval.json \
  --private-root /approved/private/v15-recovery
```

These paths illustrate user-approved private locations, not existing files or
permission granted by this document. A profile switch requires its own reviewed
plan/approval and a fresh root: the tool refuses to mix different plans. Choose
analysis initially if all36 files are actually authorized/needed; copying verified
files to a new approved root can avoid redundant network retrieval. No broad
fallback or automatic private inventory/authentication is performed.

## Failure, interruption and privacy behavior

Plan equality and exact recovery authorization are checked before private root
creation, SDK import, authentication or downloads. Storage must be user-owned,
0700, outside the repository, with no symlink parents; artifacts/state use0600.
Restrictive permissions and the approved root are enforced, not just suggested.
The operator must provide an approved durable private backup; `/tmp` is ephemeral.

A nonblocking directory lock prevents concurrent duplicate recovery. Each object
is streamed into a new private partial file. Check SHA256 before atomic promotion.
Existing valid files are rehashed/reused; corrupted files are preserved and block
overwrite. Complete interrupted partials can be promoted without redownloading;
incomplete/bad partials remain private evidence, with only unfinished files
retried. Loss of a previously complete artifact requires review, not automatic
refetch. State is persisted after every verified object; a different plan cannot
reuse that directory. Failed transfers and partial bytes consume the approved
byte budget conservatively; requests use bounded chunks and HTTP timeouts.

The transport pins API `version_label=v1`, bounds listing pagination, and requests
only each exact named file from the listing. It never calls `kernels_output`,
`kernels_logs`, datasets, uploads or submission methods. Signed HTTPS URLs stay
in memory. Every redirect host is checked **before** its request. HTTP/file-size,
missing/duplicate object, hash and provenance failures stop recovery. Explicit
no-op HTTP auth prevents requests from consulting `.netrc`, while inherited
proxy/CA settings remain intact. SDK/provider stdout/stderr/logging and raw errors
are suppressed; CLI errors/status contain no signed URLs, secrets or patient rows.
Listing responses may contain server-provided fields such as log text; the
workflow neither consumes, writes nor publishes those fields.

After recovery, verify stage-report digest, complete status, nonce, original
source commit, exact artifact map, checkpoint identity, strict reload/state
invariance/post-freeze inference, class order, role hashes and fixed aggregation/
threshold. `recovery-complete.json` means bytes/provenance recovered only;
`role_alignment_verified=false` and `v15_fitting_ready=false` remain explicit.

## Remaining original-role and fitting gates

Neither profile downloads original v4 role manifests or invents their remote
locations. They remain required in a separately reviewed recovery scope:
`train.json`, `validation.json`, `calibration.json`, with the three original
manifest SHA256s recorded in frozen-plan.json. The labeled 350-patient calibration
rows must come from the authorized original role source; V14 GPU calibration
outputs have no labels. A complete original release may include the historical
final and must not be fetched wholesale.

Prepare permitted derived rows with TRAIN labels omitted. Original/derived
manifest hashes and the exact real V15 analysis manifest must be reviewed before
separate fitting approval. Reuse the existing V15 original-role, patient/component,
both-eye, prediction/quality-order and20-pass seed checks. Recovery never calls
`load_bundle` or `analyze` and never manufactures a fitting approval to bypass
those gates. Synthetic tests cover malformed roles/eyes/components using those
same existing validators. All predictions remain REVIEW REQUIRED.

No new scientific result is produced. Sparse H, C/A/G/H regressions, H/O FP/FN,
reused validation, in-sample calibration and independent all-eight/BRSET blockers
remain as recorded. The recovery infrastructure alone does not clear any of them.

## Verification evidence

All28 new synthetic recovery tests pass. The full local CPU suite runs305 tests:
296 pass,nine optional photographic skips,no failures/errors. Tests cover exact
selection, SHA256/missing objects, wrong versions/revisions, malformed patient/
component metadata, incomplete eyes, failed/interrupted downloads, partial reuse,
private-directory controls, duplicate prevention, credential absence, sanitized
errors, byte/redirect limits and recovery/fitting authorization separation.
No synthetic test is a private Kaggle transport or scientific result.

Both virtualenvs pass pip check. Analysis lock pins and V14 frozen controller
identities remain unchanged. Public plans select32/36 objects; the isolated
installed-SDK/CLI/request-header compatibility check passes with auth/network
disabled. The added v15-recovery CI job installs only the recovery lock and runs
that isolated check plus public plans; existing CPU jobs cover the new tests.
