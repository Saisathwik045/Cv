# Status — Phase 0 foundation

## Completed

- Inspected original Cv repository and managed environment; no source notebooks/data.
- Added src-layout package boundaries for every pipeline component, portable TOML
  settings and optional dependency groups.
- Implemented check-env (optional network/GPU gate), check-config and a declarative
  five-model plan. Future stages explicitly remain unimplemented.
- Added README, persistent AGENTS rules, project context, roadmap/acceptance criteria,
  experiment protocol, execution/artifact/resume contract, Git workflow and runner audit.
- Configured CPU-only GitHub Actions for Python 3.11 and 3.12 with no credentials.
- Local Python 3.12.14: offline editable install succeeded in workspace virtualenv;
  10 unittest cases passed; compileall, config/plan checks and git diff --check passed.
  Missing-config and unimplemented-training paths fail; require-gpu exits 1 as expected.
- Read current official Kaggle CLI docs: script push/status/output supported in
  principle. Account-backed end-to-end operation is not verified.

## Scope and blockers

No training, dataset validation, split generation, checkpointing, model inference
or comparison generation has run. Those implementations belong to later phases.
No fabricated metrics or data artifacts were added. Phase 0 code/docs are complete;
GPU execution is blocked by absent credentials/CLI, Kaggle network policy and broken
shell proxy connectivity. This runtime has no verified GPU or PyTorch installation.
The data and its actual workbook schema/support have not been inspected.

Git publication uses phase/00-foundation against verified latest main 4cbc1b6.
Local checks are verified on Python 3.12 only; the 3.11 and 3.12 GitHub matrix must
run remotely after publication. A configured workflow is not an enforced required
check; main ruleset/review settings remain account actions. The PR must remain unmerged.
Publication URL and final local/remote commit status are reported in the chat handoff.

## Next action

Review the Phase 0 PR and wait for both CI checks before merging. Configure main
protection as documented in GIT_WORKFLOW.md. Begin Phase 1 from latest main after
merge: provide original ODIR-5K including data.xlsx through an ignored local mount
or accessible private dataset reference; implement validation and freeze four
patient-separated manifests only after support/leakage checks pass.

In parallel, perform the minimal account/environment actions in GPU_RUNNER.md:
repair proxy, allow Kaggle hosts, bind KAGGLE_API_TOKEN securely and confirm account
GPU eligibility/quota. A tiny nontraining GPU smoke job belongs to the next runner
verification step, followed by status polling and artifact integrity checks. Do not
launch full training until Phase 2 prerequisites pass.
