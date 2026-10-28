# Status — Phase 1 data and runner checks

## Completed

- Reviewed Phase 0 PR #1 against every foundation acceptance criterion. All latest
  push/PR checks passed on Python 3.11/3.12; local foundation tests passed. Merged
  with expected-head protection: main ce20b940020109a4dcd37b56a106d3b11ed4ab73.
  See PHASE0_REVIEW.md. Started phase/01-data from that merged main.
- Implemented discovery, authoritative data.xlsx parsing and strict patient/image
  validation. Patient-level N,D,G,C,A,H,M,O labels stay paired with both eyes.
- Implemented deterministic multilabel patient splits, per-class support checks,
  hashed inventories, separate persistent partition files, provenance and completion
  checksums; atomic publication and verified immutable reuse.
- Implemented read-only check-runner, nontraining smoke preparation and explicit
  submission/polling/version-pinned retrieval with checksum/nonce/CUDA verification.
  Failures cannot mark GPU automation verified; raw client output is withheld.
- Local Python 3.12 checks: 34 automated tests passed, editable package installed,
  compileall, Phase 0/1 config/plan checks and git diff --check passed. CI installs
  the data extra and exercises synthetic XLSX/images plus mocked runner transport
  on Python 3.11 and 3.12. Final remote CI status is reported in the PR/chat handoff.

## Real-data status

Original ODIR-5K and data.xlsx are unavailable in the workspace/configured mount.
**Real-data validation has NOT passed.** Tests generate clearly marked synthetic
fixtures; their counts/support are not real ODIR findings. No production split
manifest, patient count, disease metric or training checkpoint is claimed.
Phase 1 code is ready for real-data acceptance once original inputs are mounted.

## GPU automation status

Unattended GPU execution is NOT verified. check-runner observed: Kaggle client not
installed; supported credential-presence checks all false; authentication,
submission, status and retrieval unverified. The enforced allowlist excludes
www.kaggle.com; all shell HTTPS probes fail connecting to managed proxy:8080.
Managed environment reports no bound secrets or outbound identity.

A GPU arithmetic-only smoke bundle was prepared under ignored artifacts, with no
submission, no training, no paid provisioning and **no GPU quota used**. Mocked
transport tests exercise success/failure transitions but are not remote evidence.
See GPU_RUNNER.md for exact account/settings actions and executable commands.

## Next action

Review Phase 1 PR and its lightweight CI; keep it unmerged pending user review.
Account-only prerequisites: mount original ODIR-5K with data.xlsx (accept source
terms if needed), configure Kaggle managed secret and network allowlist, confirm
GPU eligibility/quota. The managed proxy service must also become reachable; this
is environment/platform configuration, not something to bypass in project code.

After original data is available, run discovery, choose explicit workbook/cohort
paths and prepare-data; adjudicate flagged source problems and verify real support.
After runner access works, install the runner extra and use the prepared nontraining
smoke to verify actual submission/status/retrieval/CUDA. Start Phase 2 only from
latest merged main after real-data and runner acceptance gates pass.
