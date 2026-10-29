# Phase 3 — bounded verification and prospective development protocol

Latest main was verified at `0d5986f`: PR #8 and PR #9 are merged. This branch
preserves both implementations, the architecture and all five backbones.
A private CPU review is running as
`trustretina-bounded-retinal-review-v3-426a3327/1`, source `2da502a`, dataset version 2.
It is bounded to 30 minutes and 1,536 detailed comparisons; no GPU was submitted.

The new [development protocol](DEVELOPMENT_V3.md) uses the existing fixed-seed
component stratifier with supported links only, explicit material-case leakage
audits and one shared calibration temperature with no class-specific fitting or
same-set risk-policy fitting. Historical manifests are unchanged. Actual data
readiness and the single authorized GPU pilot remain pending verified CPU evidence.
No five-model training is authorized. This section supersedes historical statuses.

# Phase 2 CPU modules implemented — training blocked (29 October 2026 IST)

PR #7 merged as reusable Phase 1 code after corrected readiness wording and passing
checks. The original architecture/five-backbone plan is unchanged. CPU factories,
shared preprocessing, read-only data adapter, gated training, checkpoint/resume,
2-D multilabel/patient evaluation and calibration/reliability interfaces now exist.
See [PHASE2_CPU.md](PHASE2_CPU.md) for tested contracts and explicit limitations.
No models/calibration parameters were fit. No new GPU smoke or paid resources.

Historical splits remain unapproved. Independent data review and a revised,
checksum-bound protocol release are required, as are calibration support/power and
a prespecified reliability risk budget. Default fitting and final release fail
closed. This status supersedes historical acceptance claims below.

# Status — reusable Phase 1 code; experimental protocol NOT approved

## Protocol review — 29 October 2026 (IST)

This section supersedes all earlier “accepted” or “acceptance passed” wording.
The previous artifacts passed mechanical checks; they are preserved as historical
sensitivity manifests and are NOT approved for training or calibration.
The 2,638-record component is a connected component of weak dHash candidates,
not evidence that these records are duplicate patients or photographs. The prior
one-positive/one-negative floor is insufficient experimental/calibration approval.
H has one calibration positive, so disjoint H-supported calibration subdivisions
are infeasible under those manifests. No models or calibration parameters are fit.

Default Phase 1 duplicate handling is restored to reject/fail closed. Historical
conservative_patient_components_v1 is retained for reproducible sensitivity analysis.
Independent image/bridge/rejected-candidate review and a versioned protocol revision
are required before any new partition approval. Original files/labels/manifests stay
unchanged and private. Five-backbone architecture remains unchanged.

PR #7 was merged for reusable validation/grouping/checkpoint primitives after
62 tests and CI passed; experimental readiness was explicitly withheld.
Independent CPU photographic evidence now supports only one probable link and two
false positives among 302 reviewed candidates, with 232,787 candidates unresolved
across the full inventory. Weak-edge union reproduces 2,638, but is not a confirmed
duplicate component. Quarantine would leave 693 patients with only 11 H positives;
no revised experimental splits were published. See [review v2](DUPLICATE_REVIEW_V2.md).
Phase 2 CPU implementation remains a separate PR, fitting disabled.

# Status — real-data validation and grouped split acceptance passed

## Latest checkpoint — 29 October 2026 (IST)

All 3,500 original labeled patient records and 7,000 referenced eye images are
retained unchanged. Original data.xlsx remains authoritative; class order is
N,D,G,C,A,H,M,O and labels remain patient-derived, not verified eye diagnoses.
Dataset API observation: andrewmvd/ocular-disease-recognition-odir5k version 2,
ID 611716. Workbook SHA256 remains
0507270f4edad478857a300514018eabefff5fe3ce04ed3b5bd4fcefbd5f54bf.

The original validator stopped at its first 64-bit dHash match with Hamming
threshold 2. Full enumeration found 232,790 perceptual candidates, zero exact
raw-file/decoded-pixel duplicates, and 186,139 candidate pairs with different
patient label vectors. These are potential annotation conflicts, not proof of
incorrect labels. All candidates remain conservatively co-grouped: 737 connected
patient components, 44 nontrivial, largest 2,638 patient records. Both eyes and
all transitive connections stay together. No candidate was dismissed to improve
class support, and no image, label or duplicate threshold was changed.

All pairs have numerical comparison evidence. A deterministic technical extremes
sample of 70 pairs (94 distinct images) was visually reviewed on six sheets along
with original workbook metadata. It includes probable photographic copies,
visually different hash matches and low-information previews. Age/sex metadata
conflicts were retained. This is not clinical or patient-identity adjudication;
no true-duplicate prevalence is estimated from the selected sample.

Full decoded validation checkpoint: source
4ddf96236263fd3529b7346e4339e65646042a03, private CPU kernel
saisathwikregula/trustretina-duplicate-group-validation-v2-b31d192e/1.
The initial grouped greedy assignment failed H support in calibration and final
evaluation; it published no splits. H occurs in 17 independent components, so
that assignment failure did not prove grouping infeasible. The versioned v3
allocator repairs deficits with deterministic whole-component moves, preserving
donor support, the fixed seed 2026 and the original fraction targets. It uses
no model metrics or seed search. The accepted assignment needed two singleton
moves from train; their IDs remain only in the private manifest.

Accepted publication: source 3692b3765bb3a7ca265fcf71f87be72cafb4bb23,
private CPU kernel saisathwikregula/trustretina-support-floor-publication-30495e59/1.
The completed validation stage was reused only after checksum/nonce/source/
dependency/settings/fingerprint verification and rehashing every raw input.
The CPU publication run passed support, patient/component separation, deterministic
reconstruction, raw immutability, immutable reuse and all output checksum checks.
No GPU smoke, model training or paid provisioning occurred in this continuation.
Earlier 15/30-minute CPU time-limit attempts remain preserved as failed attempts.

Support below is positive/negative patient records (not eye-specific ground truth):

| Partition | Patients | Images | N | D | G | C | A | H | M | O |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2696 | 5392 | 843/1853 | 945/1751 | 134/2562 | 152/2544 | 128/2568 | 97/2599 | 101/2595 | 784/1912 |
| validation | 385 | 770 | 137/248 | 98/287 | 34/351 | 27/358 | 18/367 | 4/381 | 30/355 | 97/288 |
| calibration | 210 | 420 | 80/130 | 43/167 | 23/187 | 17/193 | 9/201 | 1/209 | 22/188 | 49/161 |
| final_evaluation | 209 | 418 | 80/129 | 42/167 | 24/185 | 16/193 | 9/200 | 1/208 | 21/188 | 49/160 |

Realized fractions are approximately 77.03%/11.00%/6.00%/5.97%. The giant
component alone exceeds the 65% train target; targets are soft under indivisible
components. Every initial one-positive/one-negative class floor passes. H has
only one positive record in calibration and final evaluation: this is not evidence
of statistical power, and the planned calibration fit/policy subdivision cannot
place an H-positive record in both disjoint subsets. Review that protocol before
Phase 3. Future subdivisions/bootstrap resampling must preserve components.

Private artifacts: artifacts/runner/duplicate-group-validation-v2 (full evidence,
pair CSV and original failed support), artifacts/runner/duplicate-pair-review
(technical review and metadata), and artifacts/runner/support-floor-publication
(accepted manifests, support, complete hashes and provenance). None is committed.
Code/configuration and synthetic tests are in PR #7; 61 local tests and Python
3.11/3.12 CI passed for the implementation. No automatic merge is authorized.

Next executable phase: review/merge this Phase 1 PR, then Phase 2 CPU preprocessing,
model/runner harness implementation and environment locking. Training remains
outside this request; protocol/power limitations must be resolved before later
calibration/evaluation claims. Historical checkpoints below are superseded.

# Status — v2 smoke verified; real-data validation blocked by near-duplicate

## Latest continuation — 29 October 2026 (IST)

Resumed existing smoke saisathwikregula/trustretina-smoke-6100700cdc2f41df9219157c21f01309/2.
No new GPU smoke was submitted. The applied restricted network policy includes
www.kaggleusercontent.com; artifacts were retrieved through supported network-enabled
execution. The fresh workspace lacked previous ignored files. The prior receipt
was recovered from its chat output, and the request was recovered from pinned remote
source. Its canonical SHA256 matches the prior receipt:
39f675f8217124bdfe2b41b9da4c81776accb10192fab830c44b2c1edbb0196e.
Remote source matches the fixed nontraining template. Artifact SHA256 is
d02cb541240df7c02bf031d737e03496f13b0235bce18b6446f65fa606b6886e;
checksum and nonce both verified. CUDA arithmetic passed on Tesla T4 with
PyTorch 2.11.0+cu128. Dataset discovery independently passed: authoritative-style
XLSX, Training Images and decoded eye samples were found beneath the nested
/kaggle/input/datasets/andrewmvd/ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K.

Continued the authorized existing prepare_data workflow in a private CPU-only,
internet-disabled Kaggle script, without training or paid resources:
saisathwikregula/trustretina-original-odir-validation-e9e075f05f81/1.
Source revision: 23ceeb646cfa1fbfedbfa26c66fb3deaadc75c1c (current main at start).
The dataset API reported version 2 (dataset ID 611716) before submission.
The validation job verified the workbook against the smoke SHA256:
0507270f4edad478857a300514018eabefff5fe3ce04ed3b5bd4fcefbd5f54bf.
Retrieved report checksum, nonce, source hash and GPU-disabled metadata verified.

The job finished, but real-data validation failed closed with DataValidationError:
"Potential near-duplicate requires adjudication before splitting".
Validation was not completed; no patient split bundle/manifests were published.
No labels were altered, no images excluded and near_duplicate_hamming remains 2.
Adjudication requires a versioned source-data decision before split publication;
Phase 2/model training remains gated. The current validator stops at the first
finding and does not expose the candidate pair, so no pair identities, full image
counts or complete duplicate audit are claimed. Do not bypass this check or retry
seeds to obtain splits.

Receipts, original remote source, metadata, reports and a complete hash manifest
are saved outside Git in artifacts/runner/dataset-smoke-v2,
artifacts/runner/real-data-validation and artifacts/runner/v2-validation-checkpoint.json.
All 42 existing local synthetic/mocked tests pass; git diff --check passes.
The historical checkpoints below are superseded by this continuation.

# Status — corrected smoke v2 complete; artifact host blocked

## Latest continuation — 29 October 2026 (IST)

Runner fix committed at bd5a934 on phase/01-kaggle-mount-discovery, based on main
 d82a875903308934c5f6dae92f29ca31d0622a0f. No credential/network setup was repeated.
Version 1's submitted metadata confirms andrewmvd/ocular-disease-recognition-odir5k.
Its failure checked only /kaggle/input/ocular-disease-recognition-odir5k and did not
inspect nested mounts. Attachment is confirmed in metadata; actual file access and
mount location cannot be established from version 1's logs alone.

The runner now recursively discovers exactly one data.xlsx and one Training Images
directory beneath the configured input root. Missing, inaccessible and ambiguous
sources fail clearly. No derived CSV/replacement download is used. Separate CUDA
and dataset checks always save a checksum/nonce-bearing report. Enum-qualified
Kaggle statuses are accepted. New-version preparation verifies a previous pinned
receipt, preserves its kernel ID, and uses a fresh nonce and separate receipt.

The one additional authorized GPU smoke was submitted to the existing kernel as:
`saisathwikregula/trustretina-smoke-6100700cdc2f41df9219157c21f01309/2`.
Kaggle reports COMPLETE. Remote metadata confirms NvidiaTeslaT4, private script,
GPU enabled, internet disabled and the original configured dataset source.
No further GPU submission was attempted. Version 1's records remain unchanged.

Artifact retrieval failed with ProxyError: proxy CONNECT HTTP 403 for the exact
host www.kaggleusercontent.com. This is a new destination-policy rejection, not
authentication failure. Logs and source/metadata remain readable through the API.
The kernel catches per-check failures so COMPLETE alone proves neither CUDA nor
dataset success. Their individual results await the existing smoke.json download;
no actual mount path, dataset version or real-data acceptance is claimed yet.

Full real-data validation and split preparation remain gated on that report.
No model training or paid resources launched. The additional GPU attempt may have
used free quota; its exact amount is unknown. Apply the artifact host allowlist
change, then resume retrieval from the v2 bundle without another push. If all smoke
checks pass, continue the authorized real-data validation/splits and preserve their
reports/manifests and input/output hashes outside Git. Existing source-data checks
and split policy remain unchanged.

The focused regression suite plus all previous tests passes (42 tests).
Compilation, configuration/plan checks and git diff --check also pass. These local
checks are synthetic/mocked evidence, not remote dataset acceptance.

Saved ignored evidence: artifacts/runner/dataset-smoke-v2 (request, durable receipt,
submitted source, metadata, execution log, output listing and download diagnostic).
Historical checkpoints follow and are superseded by this section.

# Status — authenticated; single GPU smoke failed dataset mount check

## Latest checkpoint — 29 October 2026 (IST)

Resumed from main d82a875903308934c5f6dae92f29ca31d0622a0f using supported
network-enabled execution. Current managed observations confirm enforced restricted
policy including api.kaggle.com and www.kaggle.com, with KAGGLE_API_TOKEN ready.
Kaggle 2.2.5 authenticated successfully as saisathwikregula without printing
credentials. All 14 account kernels were inspected before submission; none was a
TrustRetina smoke. No local receipts/bundles were present.

Exactly one private nontraining GPU smoke was submitted:
`saisathwikregula/trustretina-smoke-6100700cdc2f41df9219157c21f01309/1`.
Submission intent and version are preserved in ignored artifacts/runner/dataset-smoke.
Remote metadata confirms the configured ODIR source, NvidiaTeslaT4, GPU enabled
and internet disabled. The remote kernel ended with KernelWorkerStatus.ERROR.
Retrieved execution logs report `RuntimeError: Attached dataset mount is missing`
at the expected /kaggle/input/ocular-disease-recognition-odir5k path. The actual
mount location was not observed. Metadata attachment alone does not verify files.

CUDA arithmetic precedes the failing dataset check in the retrieved source, but
no smoke.json/checksum was produced and the output file listing is empty. Full
GPU/dataset acceptance remains false. GPU quota may have been consumed; its
amount is unknown. This is a remote runtime failure, not a network rejection or
authentication failure. No additional blocked hostname was observed.

The installed client prints enum-qualified statuses (KernelWorkerStatus.RUNNING).
The runner now accepts that format as well as its previous bare format; a
regression test verifies resuming a pinned version without another push. All 39
synthetic/mocked tests pass. Polling resumed version 1 only; no replacement or
duplicate job was submitted. The smoke bundle/source remains immutable.

Authentication, policy, receipt, remote status/logs/source/metadata and checksums
are preserved outside Git in ignored artifacts/runner. Full real-data validation
and split preparation were not run because the smoke failed. No manifests or
real-data validation reports exist. No training or paid resources were launched.
No new PR was created. The historical checkpoint below is superseded by this one.

Do not rerun or submit a replacement automatically. A future execution requires
resolving the mount assumption and authorization for an additional smoke after
this single consumed attempt; preserve the existing receipt/version as evidence.

## Current continuation — 29 October 2026 (IST)

TrustRetina is Saisathwik045/Cv at /workspace/Cv. The checkout was clean and
synced from origin/main 7ab736131467f561201eee59ba1322e35bc7a52b, including
merged PRs #1–#4. Work continues on phase/01-kaggle-execution. Completed phases
were not recreated; the unrelated compiler repository was not modified.

## Connectivity correction

The managed proxy works. Earlier curl exit 7 / HTTP 000 results came from
network restrictions in the default command sandbox; they did not establish a
broken proxy service. A diagnostic socket connection returned EPERM by default,
then succeeded with supported network-enabled execution. The unauthenticated
www.kaggle.com request returned HTTP 401, which did not test the bound token.

Use exec_command with sandbox_permissions="with_additional_permissions" and
additional_permissions.network.enabled=true for network operations, subject to
its permission requirements. Keep inherited proxies, CA trust and TLS verification.
This mechanism permits the command to use the managed proxy; it does not override
the managed destination allowlist. No proxy bypass or policy change was attempted.

## Actual results

- Git fetch succeeded through the managed proxy; local main is synced.
- Python 3.12 virtualenv /workspace/.environment/trustretina installed the editable
  data/runner extras successfully, including Kaggle 2.2.5 and kagglesdk 0.1.38.
- Kaggle initially tried to create /home/agent/.config/kaggle, which is read-only.
  Setting supported KAGGLE_CONFIG_DIR=/workspace/.environment/trustretina-kaggle
  resolved that local import issue. No token was copied into a config file.
- KAGGLE_API_TOKEN is ready/present. The client attempted supported token
  introspection, but its production endpoint is https://api.kaggle.com.
  The applied allowlist includes www.kaggle.com and excludes api.kaggle.com.
- The client received ProxyError. A separate bounded, unauthenticated request to
  api.kaggle.com confirmed proxy CONNECT HTTP 403 (curl exit 56, destination
  HTTP 000). This is a destination-policy denial before a Kaggle auth response;
  neither token validity nor actual account username is established.
- No local verification.json, kernel-metadata.json or smoke.json receipts were
  found under the checkout/shared/scratch locations. Remote job inspection remains
  blocked. No submission was attempted and no job ID/version exists for this run.
- CUDA execution, data.xlsx/original-image attachment, remote status polling,
  checksum-verified retrieval, full validation and patient splits remain unverified.
  No training, paid provisioning or GPU quota use occurred in this continuation.

Structured observations, environment report and installed dependency versions are
preserved under ignored artifacts/runner/. No credentials or patient data are
committed. The configured direct source remains
andrewmvd/ocular-disease-recognition-odir5k.

## Completed implementation and remaining gates

Phase 0 foundation and Phase 1 validation/split/smoke implementation are merged.
Previous synthetic tests and passing CI establish implementation behavior, not
real-data or GPU acceptance. No labels, duplicate rules or support floors changed.
There are no retrieved real-data reports, manifests, dataset version or data hashes.

## Local validation

On the installed Python 3.12 environment, all 38 synthetic/mocked tests passed.
Compilation, Phase 0/1 configuration checks, Phase 0 declarative-plan check,
local environment report and git diff --check passed. No remote GPU/data
acceptance is implied by these local checks.

## Next executable step

Add api.kaggle.com to the managed environment's allowed HTTP hosts and apply the
configuration. Keep www.kaggle.com, restricted networking and the existing token
binding. Do not rotate the untested token or request proxy infrastructure repair.
No additional artifact hosts are established yet.

Then use network-enabled execution to authenticate and establish the actual Kaggle
username; inspect remote jobs and local receipts before any submission. Resume an
existing smoke if present, otherwise execute the single already-authorized short
nontraining GPU smoke. Persist its ID/version before polling. Only on success run
existing full real-data validation and patient-separated split preparation remotely,
retrieving reports/manifests with dataset/kernel versions, source revision and
hashes outside Git. Report validation failures without altering data or safeguards.
Phase 2 remains gated on those acceptance results; do not start model training.
