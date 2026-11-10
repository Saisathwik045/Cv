> Historical PR26 planning status before CPU-readiness authorization.

# V14 next-stage planning — no fitting authorized

PR #25 is merged on main `fe808b24c438503ba89462a3ed206fb7f3ae484d`.
The six-fit v13 study, its checksums and original demo remain unchanged.
[Current plan](../results/candidate-v14/README.md) · [candidate protocol](CANDIDATE_FREEZE_V14.md) · [independent dataset review](INDEPENDENT_DATASETS_V14.md).

Actual read-only verification: all six pinned Kaggle versions COMPLETE; local
source/nonce/report/output hashes pass, fresh remote report hashes match, and all
12 best/last checkpoints pass remote range-access/total-length checks. Large
checkpoint bodies were rehashed locally, not redownloaded. [Evidence](../results/candidate-v14/recovery-verification.json).
No preparation, duplicate review, split generation, training, evaluation replay,
calibration/threshold fitting, new image acquisition/upload or paid resource used.

320 ConvNeXt-Tiny ratio BCE remains the leading development candidate. V13 mean
macro AUROC 0.83273→0.85108, macro F1 0.53758→0.57156. Preserve the limitations:
H FP 25→76/F1 0.2308→0.2125; O FP 439→720; D/A/M FN 269→316,62→66,11→19.
H recall fell in two folds. Original [all-eight report](../results/resolution-v13/DISCUSSION.md)
and [v13 status](history/STATUS-through-v13-complete.md) remain the result authority.

No verified independent all-eight paired-eye cohort was identified among BRSET,
PAPILA, RFMiD, mBRSET and EyePACS. BRSET is the recommended partial-label feasibility
candidate; licensed access, actual pairs and native-endpoint compatibility remain
unverified. Unknown labels are masked, never negative. H/O and D semantics need
source guidance; partial studies cannot be called full validation. No external
patient files were downloaded or non-overlap claims established.

Proposed future runs, not approved: `v14_candidate_res320_seed2026` and
`v14_reference_res224_seed2026`, fresh ImageNet, frozen TRAIN 2,275/validation 525;
350 calibration patients only after checkpoint freeze; historical final excluded.
Same 25-epoch cosine/ratio-BCE settings and validation macro AUROC selection;
no threshold/policy fit. A new shared T per checkpoint would use calibration only,
never the old model's T. The original demo/calibration/thresholds remain unchanged;
REVIEW REQUIRED stays default.

Estimate 3.20 free GPU hours with allowance; propose strict 14,400s/four-hour cap,
max 3 sessions including one optional compatible resume. Initial reservations
7,800+4,200=12,000s; resume≤2,400s including overhead. Read-only observed free
quota 14,673.105s, reserved 0, paid disabled. No quota reserved or GPU job submitted.
Recheck actual counters, accelerator and full cap before any future approval/execution.

Next executable preparation: resolve permitted BRSET metadata/endpoint feasibility;
recover existing 224 role inputs and construct only missing native 320 validation/
calibration inputs (proposed CPU-only max 3,600s/one session), then implement/test
and hash-freeze the new worker/registry. Current local disk ~276MiB free is
insufficient; free private scratch/streaming recovery must be verified first.
New input/worker hashes, external cohort/label agreement and execution approval
are explicit blockers. No all-data refit or model promotion follows automatically.

Seven focused plan guard tests and compile/diff checks pass. The plan checker
returns valid proposal=true, execution_ready=false, GPU jobs authorized=0.
Private resume: artifacts/candidate-v14/RESUME.md and TASK_RESUME.json.
No background monitoring. [PR #26](https://github.com/Saisathwik045/Cv/pull/26) is ready for review, not merged.
All seven CI jobs passed on planning commit43c72e2 in
[run42786433487](https://github.com/Saisathwik045/Cv/actions/runs/42786433487).
This final status-only update leaves the reviewed proposal/source snapshot unchanged.
