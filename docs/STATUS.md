# Original frozen demo CPU performance — operational measurement

Continued from merged PR #37/main `f249858` on `phase/20-cpu-performance`.
Adds a separately invoked, explicitly gated Linux CPU benchmark using only the
original verified inference-v8 bundle and generated geometric non-patient images.
The bounded workload is one load, four predictions (first plus three repeated)
and one paired D-target explanation. Networking is disabled and the process has
a 180-second deadline. Fresh owner-only aggregate output stays outside Git;
predictions, input images and explanation arrays are not saved. All five bundle
hashes are verified before and after, and model state/modes must remain frozen.
No clinical evaluation, patient-data access, training, fitting, new threshold,
GPU job or demo-model substitution is introduced.

The authorized real run passed with the original CPU model: load 0.806835 s,
first paired prediction 0.208478 s, three repeated predictions median 0.151457 s
(range 0.144690–0.174269 s), paired D Grad-CAM 0.174469 s. Process-lifetime peak
RSS reached 811,536,384 bytes (773.94 MiB), not isolated model memory or a RAM
requirement. Two intra-op threads, one inter-op thread, Linux CPU-only runtime;
cgroup quota is two CPU equivalents and memory limit 8 GiB. Read the
[full performance report](CPU_PERFORMANCE.md) and
[reviewed aggregate](../results/demo-cpu-performance/measurement.json) for exact
samples, hashes, measured source commit, scope and limitations. No patient data,
images, prediction arrays, private paths or secrets are published. The development
architecture is implemented; independent clinical validation remains incomplete.
These timings establish neither clinical performance nor browser/Mac/concurrency
latency, cold-cache behavior, tail percentiles or service-level guarantees.

Verification: 374 isolated CPU-suite tests passed without skips/failures/errors;
the final four benchmark tests additionally passed, including public observation/
harness hashes and documented timings. Network and private research access were
blocked during the suite; no real benchmark runs in CI. All 223 checked local
documentation links resolve. Pip consistency, compilation, environment/config/
declarative-plan and whitespace checks passed. Frozen scientific source/configs,
dependencies and prior result artifacts remain unchanged. Remote CI is tracked
in the phase PR.

---

# Examiner rehearsal and academic assessment pack

Continued from merged PR #36/main `57c7827` on
`phase/19-examiner-rehearsal`. Adds a public-only **Examiner rehearsal** workspace
with six scenario questions, explicit unanswered/revisit/aligned feedback,
repository evidence references, session-scoped reset and optional feedback export.
This is teaching-key practice, not a grade, proof of mastery/authorship, scientific
evaluation or clinical validation. No model, dataset or private research record is
accessed by this feature; no inference, training, fitting or scientific result
computation is added.

The [examiner guide](EXAMINER_GUIDE.md) proposes an adaptable assessment rubric,
ten-minute evidence-led defence, failure rehearsal and submission/sign-off map.
It distinguishes implemented development research from independent confirmation,
truthful student contributions, institutional formatting and durable preservation.
Existing architecture/results/viva materials are reused rather than inventing
new scientific findings. Cloud-only presentation remains supported; native Mac
rehearsal is optional for that workflow. All decisions remain REVIEW REQUIRED.
Frozen source, weights, thresholds, calibration, protocols and result artifacts
are unchanged. Verification: **371 CPU tests passed**, zero skips/failures/errors,
from an isolated source snapshot with networking/private research storage blocked.
The new Streamlit AppTest exercises answer feedback, current-choice updates,
export controls and reset isolation, with the model constructor forbidden.
This is synthetic UI testing, not a new rendered-browser or real-model run.
All 217 checked local documentation links resolve. Compilation, pip consistency,
phase0 configuration/declarative-plan checks, environment probes and whitespace
checks passed. Remote CI outcomes are recorded in the phase PR.

---

# Guided cloud demo and local readiness — cloud-only workflow

Continued from merged PR #34/main `7861da9` on
`phase/18-cloud-demo-readiness`. The user chose cloud-only operation and approved
the guided synthetic demonstration plus cloud readiness recommendation. This
branch is independent of the optional, unmerged Mac recovery work.

Adds a generated-image guided workspace: explicit CPU predict/initial Grad-CAM,
eight original-model probabilities, REVIEW REQUIRED, experimental signals,
synthetic-context JSON export, selectable explanation, replacement invalidation
and scoped reset. Images are generated in session, never read from patient
storage or committed. The original manual predictor and scientific source remain
unchanged; V14/V15 results are not substituted for the original demo.

Adds on-demand hash-only bundle checks, exact dependency pins, public-evidence
hashes, bounded nonredirecting loopback health and server-protection checks. The
snapshot distinguishes local preflight from unverified external viewing, model
execution not run by readiness, unavailable uptime guarantees and incomplete
clinical validation. No automatic recovery, tunnel, public deployment, training,
calibration, private patient-data access or Kaggle job is introduced. See
[the cloud demonstration guide](CLOUD_DEMONSTRATION.md).

Verification: **368 CPU tests passed**, zero skips/failures/errors, from an
isolated source snapshot with networking and private research storage blocked.
All 204 checked local documentation links resolve; dependency consistency and
whitespace checks pass. Live Chromium inspection observed local preflight PASS
(all five bundle identities, 50 dependency pins and public evidence), while external
viewing remains NOT VERIFIED. One explicitly triggered original-model CPU
prediction on the generated pair and initial D-target Grad-CAM completed. All
eight outputs, both explanation overlays, both JSON downloads, replacement
invalidation and reset were inspected; downloaded exports were verified privately.
This is operational evidence only, not a medical or V15 research evaluation.
Remote CI outcomes are tracked in the feature PR. The loopback preview remains
running in the current runtime; availability ends if the environment stops.

---

# Academic evidence hub and research defence — reporting extension

Continued from merged PR #33/main `60d0957` on
`phase/16-academic-evidence-hub`. Adds the public-only **Research study & viva**
dashboard workspace with research-question mapping, both V15 temperatures,
recorded NLL/Brier/ECE, all-eight-class errors/supports, fixed risk-ranking
strategies, reviewed reliability figures, critical analysis and viva questions.
The aggregate JSON and figures are checksum verified before display. Missing or
tampered evidence fails closed. No original inference implementation, checkpoint,
preprocessing, calibration, threshold, research protocol or recorded output changed.
Presentation code lives in `dashboard`, outside the frozen all-`src` V14 worker
snapshot; the original scientific source-integrity guard remains unchanged.

The [research defence chapter](ACADEMIC_RESEARCH_DEFENCE.md) adds an evidence
matrix, design justification, ablation/decision trail, failure hypotheses with
falsifiable follow-up, truthful contribution/AI-assistance template, reproduction
levels and deployment assessment. The [all-eight-class table](ALL_EIGHT_CLASS_COMPARISON.md)
copies public recorded values; it is not new scientific evaluation. All decisions
remain REVIEW REQUIRED. Independent validation, author sign-off, native Mac
rehearsal and durable backup remain incomplete.

The user requested a cloud preview remain running. This means an observed
loopback process in the current workspace, not permanent hosting, an external
preview URL or a public deployment. It stops when the environment stops. No
patient data, private result arrays, new fitting/inference or Kaggle job is needed
for the evidence hub. The private model bundle and prepared Mac archive are not
published or rewritten. Local verification: **355 CPU tests passed**, zero skips,
failures or errors, in an isolated source snapshot with networking and private
research-storage access blocked. All201 checked local documentation links resolve;
pip consistency and whitespace checks pass. The research-demo CI matrix now runs
both the paired synthetic submission tests and academic-page tests. Remote CI
outcomes and rendered-browser review are recorded in the PR.
Chromium inspection observed the new navigation, question/evidence map, V15
metric view, reviewed reliability panels and viva expanders without a visible
application error. No real-model inference was run for this reporting extension.

---

# Final academic submission documentation — development research through V15

Continued from merged PR #32/main `dd8f281` on
`phase/15-final-academic-submission`. The [final research report](FINAL_RESEARCH_REPORT.md),
[phase audit](PHASE_COMPLETION_AUDIT.md), [presentation outline](FINAL_PRESENTATION_OUTLINE.md)
and [submission/reproduction checklist](FINAL_SUBMISSION_CHECKLIST.md) consolidate
public committed evidence only. Architecture, datasets/target semantics,
preprocessing, training/model comparisons, calibration, evaluation, MC/quality,
Grad-CAM, negative findings and research limitations are documented. All phases
through V15 are audited without opening private patient data or replaying the
historical final cohort. No research fitting, inference, training or GPU job ran.

The original demo remains the v5/Phase6 frozen candidate, separate from later
V14/V15 results. Three new synthetic Streamlit tests use generated PNGs and a fake
service for paired/invalid-input gating, eight-class presentation, REVIEW REQUIRED,
download-control presence, explanation-target selection, reset and bundle failure.
They do not load private weights or establish clinical performance, external browser
download/Cloud access or native Mac runtime. Four documentation/source consistency
tests verify the public report, phase coverage, source digests and links. Stale
README implementation claims and Mac branch instructions were corrected.

Local verification: **350 CPU tests run, 350 passed, zero skips/failures/errors**.
Optional photographic dependencies were installed in an isolated temporary target;
the original app/report environment still matches all 61 unique lock pins and
passes pip check. Test credentials were removed, CUDA hidden, networking disabled,
and access to the private shared research root denied by the test harness.
Frozen V14 identity, configuration/declarative-plan/v3/v4, shell syntax, compilation,
sterile recovery/public-plan and whitespace checks pass. Recorded evidence is in
[the submission verification record](../results/submission-audit/verification.json);
remote CI outcomes are tracked by this PR. Synthetic fixtures are software checks,
not repeated real scientific execution.

**Ready for academic review as development research**, conditional on current
checks/reviewer approval and student-supplied author/institution/supervisor details,
institutional PDF/slides formatting, citation review and sign-off. Independent
all-eight clinical confirmation, a supported clinical acceptance policy, native
Mac hands-on verification and durable encrypted private backup remain incomplete.
BRSET access/DUA/label compatibility and prospective cohort/support/overlap design
remain separate blockers. No private artifacts are published; no automatic merge,
model/demo promotion, backup transfer or new external evaluation is authorized.
All research decisions remain **REVIEW REQUIRED**.

---

# V15 real CPU calibration and reliability — completed development analysis

Continued from merged PR #31/main
`b60ccf81044c2904f82e6a40e11efc4728c4bf5a` on
`phase/15-public-research-report`. The separately authorized real V15 CPU analysis
completed once for both frozen V14 candidates. This documentation task publishes
only [verified aggregate results and reviewed reliability diagrams](../results/calibration-v15/RESULTS.md);
it does not repeat fitting, inference or training. Prospective protocols, scientific
code, checkpoints, original calibration/thresholds, historical results and demo
remain unchanged. All research decisions remain **REVIEW REQUIRED**.

All three reconstructed original V4 manifest hashes and all 36 restored V14
artifact hashes/provenance passed. Original/derived patient, both-eye, role,
component and class alignment, saved 20-pass MC seeds/order and resolution-specific
technical quality inputs passed. One component-equal shared temperature per
checkpoint was fitted on 350 calibration patients / 331 components:
**320 px T=1.105496125613; 224 px T=1.002532561130**. Both optimizers succeeded
without boundary solutions. All p>=0.5 binary decisions and classification metrics
remained unchanged. Calibration is in sample; the 525-patient / 518-component
validation cohort was reused for selection and supports descriptive results only.

NLL and Brier decreased slightly; ten-bin ECE increased in both roles/resolutions.
Shared NLL optimization did not establish uniform class calibration improvement.
The fixed quality factor worsened Hamming risk-coverage AURC throughout; all
prespecified strategies and class regressions are retained. Paired 320-minus-224
AURC intervals from 200 whole-component bootstrap replicates all include zero;
they are conditional descriptive intervals, not independent confirmation.

The public report and curated aggregate JSON were checked against the recorded
private aggregate source, detailed/aggregate output digests and provenance.
Four original saved reliability PNGs were reviewed visually and for metadata,
then copied byte-for-byte; no patient records, row predictions, original manifests,
private inventories, approvals, checkpoints, credentials or operational storage
paths are published. Public consistency tests check data schema, negative findings,
table values, frozen identities, links and reviewed figure digests. Their success
does not independently authenticate unavailable private data.

Independent validation remains incomplete: a prospectively registered untouched
cohort/endpoint, compatible authoritative all-eight labels, adequate class/component
support and required data-access/ethics/DUA approval remain necessary. BRSET
credentialing/CITI/DUA and label compatibility remain unresolved. No new inference,
historical-final access, GPU job, deployment or further fitting is authorized here.

Private preservation assessment verified 98 research files and 384 execution-source
files. No verified durable encrypted backup exists; destination/recipient approval,
encrypted transfer, a full-hash authenticated restore drill and a second durable
copy remain outstanding. No private backup payload is committed or transferred by
this documentation update.

Validation: all nine new public-report tests pass. The isolated CPU source-snapshot
suite ran 343 tests: 334 passed, nine optional photographic tests skipped, zero
failures/errors; CUDA was hidden, credentials removed and test networking disabled.
Every exported number was compared with its recorded aggregate source. Frozen V14
controller identity, package consistency, configuration/declarative-plan/v3/v4,
compilation and whitespace checks pass. Remote CI outcomes are tracked in this
documentation PR; local fixtures are software checks, not additional real analysis.

---

## Historical implementation and readiness records

The following sections preserve the scope and observations of earlier software
PRs. Their statements about unperformed recovery or real fitting refer to those
earlier tasks; the completed real execution above supersedes them as current status.

# V15 original manifest reconciliation — synthetic software only

Continued from merged PR #30/main `6c33e8c` on
`phase/15-manifest-reconciliation`. The isolated stdlib-only offline utility and
[reconciliation protocol](V15_MANIFEST_RECONCILIATION.md) implement exact V4
publisher serialization, all-three source hash checks before parsing, strict
patient/eye/label/component/role validation, original SHA256 acceptance, private
permissions, record-free diagnostics and atomic three-file publication with disk
readback hashing. Public planning binds the full authenticated V13/V6 versions,
source hashes, original hashes and source/provenance identities. No private
inputs, credentials, original manifests or historical final were recovered or
opened; no real reconstruction, jobs, inference, fitting or training occurred.
The previously recovered 36 V14 artifacts were not accessed by this workflow.
V15 loader guards, frozen V14 code/artifacts/configuration/thresholds and demo
remain unchanged. Frozen V14 controller identity verification passes.

Validation: 29 new synthetic tests, including byte identity against the actual
V4 publisher, source/output mismatch refusal, row order, duplicate patients,
missing eyes, class order, labels, component membership/mixed labels, roles,
missing inputs, symlinks, private permissions, failed writes and staged corruption.
Full CPU source-snapshot suite: 334 run, 325 passed, nine optional photographic
skips; CUDA hidden and test-process networking disabled. Configuration, declarative
plan, v3/v4 checks, pip check, compilation and whitespace checks pass. Remote CI
outcomes are tracked in the reconciliation PR. Synthetic values are not research
results or proof of actual private source accessibility/hash reproduction.

The method is technically ready for separately authorized verification; real
success remains unproven. Next permission must explicitly authorize selective
credential-backed recovery of only the three pinned derived-row objects, including
patient-derived labels, and CPU reconstruction into named private storage under
the reviewed plan/code hash. No approval is generated here. V15 analysis/fitting
requires a further independent approval and verified real analysis manifest.

---

# V15 selective recovery infrastructure — preparation only

Continued from merged PR #29/main `e29c4641d17ffb5cafd9e1c8af5c29abb20744f6`
on `phase/15-recovery-readiness`. [Recovery protocol/readiness](V15_RECOVERY_READINESS.md)
adds public planning and separately approved, selective CPU recovery for the two
pinned V14 output versions. No private recovery or credential access occurred.

The existing analysis `.venv` is preserved. A separate Python 3.12 recovery
virtualenv at `/workspace/.venvs/trustretina-recovery` installs the unchanged
artifact-recovery.lock exactly: Kaggle2.2.5/kagglesdk0.1.38, pip check passed.
SDK imports/CLI help/version-label compatibility passed in an empty child
HOME/config with authentication replaced before package initialization and
network disabled. Public Kaggle web200/API401/root404, PyPI200 and GitHub200
confirm connectivity only; account/private-version/redirect access is unverified.

Default metadata profile selects32 exact files and excludes checkpoint binaries.
Analysis profile selects36 files, including the four opaque best/last checkpoint
bytes required by the unchanged V15 validator. Neither profile includes datasets,
image tensors, ImageNet weights, final data or logs. Public inventory/nonce/source/
report/artifact SHA256 pins, exact recovery approval, approved private0700 root,
private0600 partial/state files, locks, bounded selective HTTP transfers,
interruption recovery, rehash/reuse and sanitized errors are implemented.
Recovery never creates or accepts fitting authorization. Original role-manifest
and label recovery remain separate, source-verified blockers; completion never
asserts patient alignment or V15 fitting readiness.

Validation: all28 new synthetic recovery tests passed. Full local CPU suite:
305 run,296 passed,nine optional photographic skips,zero failures/errors. Tests
ran in a temporary source snapshot with CUDA hidden and test-process networking
disabled. Both Python environments pass pip check; analysis CPU/app/report locks
still match. Isolated installed-SDK/CLI/no-netrc checks, public metadata/analysis
plans, compilation, configuration/declarative-plan/v3/v4 checks and whitespace
checks passed. Frozen V14 controller identity verification still passes. An
additional credential-free recovery CI job validates installation and SDK/CLI
compatibility; actual remote outcomes will be recorded in the PR. No test values
are research results. No Kaggle jobs, real fitting/inference,
private files or credentials were accessed. Frozen demo, prior results,
thresholds, models, old source and dependency locks remain unchanged.

---

# V15 calibration/reliability software — synthetic verification only

V15 source development starts from remote main
`badbf8af8ef3c0ac57a942f70e594e9b8634a419`, on
`phase/15-calibration-reliability`. [Prospective protocol](CALIBRATION_RELIABILITY_V15.md)
and [execution readiness](V15_EXECUTION_READINESS.md) define the separate real-data
permission boundary. No real V15 scientific result is reported.

Implemented CPU analysis for both frozen V14 candidates: trusted artifact/checkpoint
SHA256 checks, original role-manifest and patient/component/eye/class alignment,
separately gated shared-temperature fitting, NLL/Brier/10-bin ECE, per-class
reliability diagrams/counts/support, saved 20-pass head-only MC and TRAIN-only
resolution-specific technical quality, paired FP/FN/precision/recall/F1 differences
and descriptive tied-score risk-coverage/component intervals. The unchanged shared
algorithm is reused; no threshold or acceptance policy is fitted. All predictions
remain REVIEW REQUIRED. No historical final access, demo change, private dataset
or checkpoint access, model inference, Kaggle submission or real-data fitting.

Validation: all 28 V15 synthetic tests passed. The complete CPU suite ran 277
tests: 268 passed, nine optional photographic tests skipped, no failures/errors.
This includes existing toy optimizer/resume fixtures, with no real-model training.
Tests ran from a temporary source snapshot with CUDA hidden and test-process
network transport disabled. Readiness/synthetic CLI, phase0/phase1 config,
declarative plan, v3/v4 development checks, environment report, compilation,
`pip check`, and `git diff --check` passed. Original V14 controller freeze
verification passes byte-for-byte; `src/`, old scripts, configs, requirements,
results and demo remain unchanged. PR CI outcomes are tracked in
[PR #29](https://github.com/Saisathwik045/Cv/pull/29). Synthetic fixture values
are not research results.
Real-data execution remains blocked on separate explicit authorization, a reviewed
hash-bound manifest and verified permitted private artifacts/role records. BRSET
access and independent all-eight compatibility blockers persist. Existing optional
photographic dependencies and Kaggle recovery tooling remain absent locally.

---

# V14 two frozen fits complete — reused-validation development only

Merged PR27/main3b07a7634f3e6530d66b2ec41c70d837893992c6; approved plan/source/config/
protocol/dependency/input identities preserved. No preparation, duplicate review
or v13 training was repeated. Exact two fits completed sequentially with one
session each, zero failures/resumes, and no paid scaling.

| Fit | Epochs / best | Macro AUROC | Macro AP | Macro F1 | Micro F1 |
|---|---:|---:|---:|---:|---:|
| ConvNeXt320 | 11 / 5 | 0.8596 | 0.5993 | 0.5371 | 0.5532 |
| ConvNeXt224 | 11 / 5 | 0.8480 | 0.5812 | 0.5332 | 0.5290 |

[Complete all-eight precision/recall/F1 and TP/FP/FN](../results/candidate-v14/RESULTS.md)
and [CSV](../results/candidate-v14/validation-per-class.csv).320 leads aggregate
validation metrics but is not uniformly better: H FP6 vs21 but FN12 vs11; O FP119
vs118, FN53 vs61; D/A/M misses107/14/4 vs130/7/5. C FP99 vs30 and F1 0.3827 vs 0.6222;
A recall0.44 vs0.72; G recall0.50 vs0.5625. Exact match0.2990 vs0.3143. No independent
confirmation or clinical superiority; no automatic promotion/refit/demo change.

Private pinned jobs:
- saisathwikregula/trustretina-v14-fit-320-7bbe5ade/1
- saisathwikregula/trustretina-v14-fit-224-779a38cd/1

All 34 output-file hashes passed, including best/last checkpoints, aligned raw eye/
patient predictions, histories, dependency records,20-pass seeds and quality arrays.
Both strict best reload/state-invariance/post-freeze inference checks passed.
Small artifact/prediction recovery was independently verified from both versions;
large checkpoints and patient records remain private outside Git.
[Artifact hashes](../results/candidate-v14/verified-artifact-registry.json),
[fresh retrieval](../results/candidate-v14/fresh-retrieval-proof.json),
[run registry/receipts](../results/candidate-v14/frozen-plan.json).

[Actual quota accounting](../results/candidate-v14/gpu-time.json):3826.407s(1.062891h)
observed account delta vs14400s cap; two sessions vs maximum3, no resume. Baseline
timeUsed6926.895s and refresh2026-11-14T00:00Z retained throughout. Final used10753.302s,
free10846.698s, reserved0, paid disabled. Worker sum3767.781s is distinct from billed
quota. Internal conservative reservations12000s. Foreground queue driver exited;
no background monitoring or further GPU session is active.

No historical final access, external evaluation, temperature/threshold fitting or
working-demo modification. Calibration output metadata has no labels; inference
ran only after checkpoint freeze/strict reload. REVIEW REQUIRED stays unchanged.

Next executable research step requires a separately authorized, prospectively
recorded calibration/reliability analysis for these new checkpoints, using saved
post-freeze inputs/predictions. Do not reuse the original demo's temperature or
threshold experiments. BRSET remains a separate credentialing/CITI/personal-DUA/
label-compatibility blocker; no agreement/form submitted. Independent all-eight
confirmation remains unavailable. GPU training queue is complete; do not resubmit.
