# GPU runner research — 28 October 2026

## Verdict

**Kaggle is a suitable first candidate, but unattended GPU execution from this
Codex session is NOT verified.** No job was submitted, no training started and no
paid resource provisioned. API feature support is documented; account entitlement,
quota and end-to-end submission/status/artifact retrieval remain untested.

## Evidence and sources

The former Kaggle/kaggle-api repository resolves to Kaggle/kaggle-cli. Official
upstream documentation was read using the GitHub connector:

- [CLI authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)
- [Kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [Kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)

Documented operations support noninteractive Python scripts (`kernel_type=script`),
GPU metadata (`enable_gpu`, `machine_shape`) and dataset/output attachments.
`kaggle kernels push -p <folder>` uploads **and starts execution** by default;
`--no-run` saves a version only. `kaggle kernels status <owner/slug>` polls;
`kaggle kernels files <owner/slug/version>` lists outputs and
`kaggle kernels output <owner/slug/version> -p <folder>` retrieves artifacts.
These are upstream capabilities, not implemented project commands or a tested run.
The docs list accelerator IDs including NvidiaTeslaT4 and NvidiaL4, with explicit
account/competition restrictions. Do not assume any listed accelerator is available.
No fixed weekly quota, session cap, storage limit or queue availability is claimed.

## Current blockers

Kaggle CLI is absent, KAGGLE_API_TOKEN/legacy credentials are not configured, and
no managed secrets are bound. The runtime's enforced network allowlist excludes
www.kaggle.com; additionally all shell HTTPS probes currently fail at proxy:8080,
even to allowed hosts. Direct authenticated Kaggle calls cannot be tested.
The GitHub connector cannot proxy Kaggle APIs. Account verification, accelerator
eligibility, remaining quota, ODIR access/license acceptance, output-download domains
and cross-session checkpoint durability are unverified.

## Minimal next account/environment actions

1. Repair the managed runtime's proxy connectivity and allow `www.kaggle.com`
   (plus API/download hosts actually needed by the installed client, as reported by
   policy diagnostics). Keep TLS verification and the managed proxy; do not bypass it.
2. In [Kaggle API settings](https://www.kaggle.com/settings/api), generate a token and
   bind it as a managed secret named `KAGGLE_API_TOKEN`. Do not paste it into chat,
   source files, GitHub issues or a committed config. The official CLI also supports
   OAuth, `~/.kaggle/access_token` and legacy kaggle.json. Presence alone is not auth.
3. Check the Kaggle account's GPU eligibility/verification prompts, remaining GPU
   quota, available accelerators, maximum session time and output limits in the UI.
   Accept the original ODIR dataset's terms if needed; provide its exact accessible
   dataset reference or mount original files including data.xlsx privately. Do not
   replace it with a processed CSV dataset.
4. After Phase 0, install the runner extra, verify CLI version/help and run read-only
   `kaggle kernels list --mine`. Then authorize a tiny, nontraining GPU smoke job
   that records torch/CUDA/device details and an artifact checksum; poll to completion,
   download the exact version's outputs and verify them. Only this can establish
   working submission/status/retrieval and GPU access for this account.

A future adapter will package a pinned source revision/wheel, config and frozen
manifest into a private script kernel; record returned kernel/version IDs; poll
with bounded backoff; fail on error/timeout; retrieve/checksum artifacts before
marking local completion. Prefer no runtime internet by attaching dependencies and
pretrained weights when feasible. Persist the pinned source, environment lock and
input/output provenance. Never assume a Kaggle interactive session or notebook cell
stays alive. Export checkpoint chunks and reattach previous outputs for continuation;
verify that behavior before scheduling five full models.

If Kaggle remains inaccessible or its quota cannot support the experiment, stop at
that evidence and ask the user to choose an account-backed alternative GPU runner.
No paid fallback has been selected or provisioned. A verified runner is a Phase 2
prerequisite, not a claim of Phase 0 completion.
