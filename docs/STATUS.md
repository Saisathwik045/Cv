# Status — Phase 1 merged; Kaggle access blocked

PR #2 was reviewed and merged after all latest checks passed. Merged main:
16f0a5c64f991a73f383bb78d8fdfe9f1740a9d1. Real-data validation is still pending.
See KAGGLE_ACCESS_STATUS.md for the current evidence and resume instructions.

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
- Local Python 3.12 checks: 38 automated tests passed, editable package installed,
  compileall, Phase 0/1 config/plan checks and git diff --check passed. CI installs
  the data extra and exercises synthetic XLSX/images plus mocked runner transport
  on Python 3.11 and 3.12. Final remote CI status is reported in the PR/chat handoff.

## Real-data status

The user supplied the Kaggle reference andrewmvd/ocular-disease-recognition-odir5k.
It is now configured for direct attachment; content access/discovery remains
unverified because connectivity and authentication are blocked.
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

A GPU arithmetic and dataset-discovery smoke bundle was prepared under ignored artifacts, with no
submission, no training, no paid provisioning and **no GPU quota used**. Mocked
transport tests exercise success/failure transitions but are not remote evidence.
See GPU_RUNNER.md for exact account/settings actions and executable commands.

## Next action

Do not start Phase 2. Apply the supported environment host/secret settings described
in KAGGLE_ACCESS_STATUS.md. No manual data download/reupload is required. Check the
applied settings once in a fresh cloud session; do not repeat failed network probes
in the unchanged session. Install/authenticate the real Kaggle client when possible,
identify the actual owner and submit the single already-authorized GPU/dataset smoke.
Resume with its saved ID rather than submitting another job. If smoke and retrieval
succeed, run existing validation/splitting remotely and retrieve manifests with
provenance. Keep raw images, patient manifests and credentials outside Git.
