> **Current: [both exact fits completed](RESULTS.md).** Validation320/224 macro AUROC0.8596/0.8480, macro F10.5371/0.5332. Two sessions,3826.407 free GPU-quota seconds; no failures/resumes. All checkpoint/prediction/history/log hashes and post-freeze checks verified. Class-specific regressions are retained, not hidden. No independent confirmation, fitting of temperatures/thresholds or demo replacement. [Artifacts](verified-artifact-registry.json) · [Accounting](gpu-time.json). CPU readiness/planning records below are historical.

# V14 candidate/freezing and independent-evaluation plan

**Research/planning only. No jobs submitted, models/temperatures/thresholds fitted,
external patient files acquired, or demo changed.** 320-pixel ConvNeXt-Tiny ratio
BCE remains the leading development candidate; the 224 reference and all six v13
results remain unchanged. REVIEW REQUIRED stays the decision for every prediction.

- [Candidate protocol, partitions, exact future matrix and metrics](../../docs/CANDIDATE_FREEZE_V14.md)
- [Authoritative dataset review, taxonomy limits and overlap gates](../../docs/INDEPENDENT_DATASETS_V14.md)
- [Machine-readable proposal](proposed-plan.json), [source catalog](dataset-options.json), [measured cost derivation](compute-estimate.json)
- [New read-only six-run recovery verification](recovery-verification.json)

## Verified state and feasible next step

PR #25 merged at `fe808b24c438503ba89462a3ed206fb7f3ae484d`; its final head is an
ancestor of this branch; PR25 head CI passed. The merge commit has no checks
returned by the legacy status connector, so its own CI is not assumed passed.
This planning branch is checked separately. All six pinned private Kaggle versions
remain COMPLETE. Existing local receipt/source/report/output SHA256 checks passed;
fresh remote report SHA256 matches and each best/last checkpoint's one-byte
retrieval/total length checked. This is not a claim of fresh full-checkpoint
download: their full bytes were rehashed in existing local copies. Nothing was
rebuilt or retrained. Remote access/retention is an observation, not a guarantee.

No compatible independent all-eight paired cohort was identified in the five
reviewed options. **Next: obtain permitted BRSET metadata and resolve native
endpoint/pairing/license feasibility, then prepare only missing native 320
validation/calibration inputs and a reviewed worker/contract.** Partial-label
transport studies cannot confirm all-eight ODIR performance. H/O source semantics
may block even a new cohort's exact label harmonization. Full confirmation needs
a suitable independently labeled paired-eye panel with documented definitions
and overlap exclusion; geography, filenames and missing labels are insufficient.

CPU readiness work is proposed at no GPU-quota cost, with at most one new
GPU-disabled preparation session capped at3,600 wall seconds. It has not run.
Existing TRAIN inputs and224 role caches are recovery inputs, not rebuild targets.
Current local free disk is about 276MiB; adequate free private scratch or a reviewed
streaming recovery route is required before new caches/checkpoints. No paid
storage, new dataset upload or access-form submission is assumed.

## Exact proposed future matrix — not approved

| Order | Run ID | Resolution | Role | Worker / reserved quota seconds |
|---|---|---:|---|---:|
|1|`v14_candidate_res320_seed2026`|320|Leading candidate|7,500 /7,800|
|2|`v14_reference_res224_seed2026`|224|Matched control|3,900 /4,200|

Both fresh ImageNet, ratio BCE, same v4 patients/containment groups, optimizer,
25-epoch cosine horizon, six-epoch meaningful-improvement early stopping, seed 2026
and mean-eye p≥.5 predictor. Validation macro AUROC selects epochs; no seed search,
outer-fold promotion or independent-result strategy selection. Calibration fitting
is a later CPU gate after checkpoint hashes freeze, separately versioned from the
old demo bundle. No threshold or acceptance-policy fit is proposed.

Full 25-epoch measured-history estimate: 2.49 worker hours, roughly 2.56 quota hours
using the historical accounting ratio, or 3.20 with 25% allowance. **Proposed strict
14,400 free GPU seconds, maximum three sessions including failures/resumes.** Two
initial reservations total 12,000s; at most one compatible resume ≤2,100 worker
seconds+300s overhead consumes the remaining 2,400s. Use actual account-counter
charges and fresh quota/accelerator checks, not worker time. Stop/incomplete if a
budget or compatibility gate fails; never blindly resubmit an ambiguous job.
No external-evaluation GPU session is included in this budget.

Observed free quota 14,673.105s, reserved 0, paid disabled, refresh reported
14 November 2026 05:30 IST. This is neither a reservation nor training approval;
recheck before any future execution. New actual role-input/worker hashes are
explicitly null, so this snapshot is **not an execution approval target** yet.

## Reproducible checks performed

Using the existing private recovery environment with securely configured Kaggle
credentials (never print them):

```sh
python scripts/check_candidate_plan.py
python -m unittest discover -s tests -p test_candidate_plan.py -v
python -m compileall -q scripts/check_candidate_plan.py scripts/verify_candidate_recovery.py tests/test_candidate_plan.py
# Read-only remote status/inventory, small reports and one byte/checkpoint:
python scripts/verify_candidate_recovery.py --remote
```

The first checker validates the **proposal**, while reporting execution_ready=false
and zero GPU jobs authorized. It does not implement training or a partial-label
evaluator. Seven focused mutation tests cover unauthorized fitting, data-role
leakage, unknown-label misuse, false full-eight claims, loss drift and resume
budget overhead. They are software guard tests, not real model/data validation.
Compilation/diff checks pass. All six artifact checks were actual authenticated
read-only retrievals. A legacy listing endpoint returned basenames/unsuitable
sizes; the final verifier used versioned session-output names and actual byte-range
responses. No missing-artifact or training failure was inferred from that endpoint
mismatch. Sandbox proxy failure was resolved using the supported network-enabled
execution route, retaining proxy/TLS protections.

Private evidence and resume state live under ignored `artifacts/candidate-v14/`.
Original v13 RESUME.md/TASK_RESUME.json/approval/registry were read and preserved.
New proposal file identities are in checksums.json. These identify this review
snapshot, not a new training authorization. No background monitoring is active.

[PR #26](https://github.com/Saisathwik045/Cv/pull/26) is ready for review. All seven CI jobs passed on planning commit43c72e2 in [run42786433487](https://github.com/Saisathwik045/Cv/actions/runs/42786433487). Execution remains blocked and unapproved.
