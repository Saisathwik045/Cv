# V14 CPU readiness completed — GPU execution awaiting approval

PR26 is merged at `f722cb51167b1694802ffdbbe297866b7e8db2a9`. The completed v13 study and its private artifacts are reused. No GPU training, real-data neural inference, temperature/threshold fit, external inference, demo change or paid resource occurred in this readiness stage.

## Ready

- Free private `/tmp/trustretina-v14-private` scratch passed a capacity/write check. Existing verified artifacts were preserved. Current free capacity exceeds the worst three-session best/last checkpoint budget plus512MiB margin. Scratch is ephemeral; pinned private outputs/receipts are the durable recovery authority.
- Actual TRAIN224/TRAIN320 were rehashed in place. Existing224 validation/calibration tensors were retrieved from their pinned v5/v6 input versions and verified; no TRAIN or224 image preparation was repeated.
- The sole authorized CPU job `saisathwikregula/trustretina-v14-inputs-c057c643/1` completed in276.103s, provider timeout3,600s, GPU/TPU disabled. All1,750 selected native image SHA256 checks passed, two320 tensors were built,224 tensors stayed byte-identical, and historical final was excluded.
- The new GPU-mounted calibration metadata contains no labels. The original labeled metadata remains separate and unchanged. Actual patient/component separation and all-eight class support passed for TRAIN2,275/2,240 groups, validation525/518 groups, calibration350/331 groups.
- Worker/registry/recovery are implemented for exactly `v14_candidate_res320_seed2026` then `v14_reference_res224_seed2026`. Same fresh ImageNet, ratio BCE, max25 cosine epochs, validation patient macro AUROC selection, mean-eye sigmoid aggregation and p≥.5. Best/last epoch checkpoints, strict reload, deterministic resume, post-freeze calibration-input gating, sample-aligned predictions and source/nonce/output-hash recovery are enforced.
- Financial guards retain14,400 free GPU-quota seconds/max3sessions, including failures/resumes/overhead. Initial reservations7,800+4,200=12,000s; the sole optional compatible resume≤2,400s. No fourth session or paid fallback. Quota baseline is created only on later authorized execution, not during readiness.

## Frozen identities

| Item | SHA256 |
|---|---|
| Plan | `3e82ea2c70524b2d7348ea4f7df8bc818535f8db95ec6ce7f15923693d4b0a18` |
| Worker/source | `485b7e05b1e7fbabea136c4c1925cfa81979dd3ad1892f796742d509c6dd06c3` |
| Configuration | `88f074d6eed6f7848765fc02d9604250d812ed569719e8ec458520fe4cce8507` |
| Controller | `b1329aab14583592b5ee9f888b47eea342270d6ca6535149f3e9fa200ecd04da` |
| Protocol | `6db9d7f040cd8921a895311df8f79b14251049772cc4e49ec13931de9db2e397` |
| New role-input index | `ddbc1f4f0b0e21fa1c6ffd3f917ee7c958fd7d8b9ee9dc13aca1d386d6a62b4a` |
| Existing TRAIN index | `47c421217bb42cace8dcb11bd463d24a2d596ab3b74e33c7f40434335d45f08d` |

Source revision: `09632a1347435991e6177cea5f5cb68ac8115655`. [Frozen registry](frozen-plan.json), [actual readiness](readiness.json), [CPU report](cpu-inputs-report.json), [private job receipt](cpu-preparation-receipt.json), [dependency contract](../../requirements/candidate-v14-environment.json). The original proposed-plan.json remains historical; current verified prepared_inputs in readiness.json replace its null readiness placeholders. No missing physical tensor is represented by metadata alone.

| Actual tensor | SHA256 |
|---|---|
| calibration224 | `9587ddf4da564ccff18cfa1fbbf19a1c28cdf0d5be58197ed9156baf8fc81aa0` |
| calibration320 | `32c59551fa33fc27a2cbb3ba14f00b3af3d657485422363865219bc3e29b0a56` |
| train224 | `274679427634724590dd9a7879cd562879e3d49dea6fdf62aa4e84a7d51fc70f` |
| train320 | `9cd180cc9fff089b87320ad1a6f097c8b632a277e41dee5b54e06c9ecb0bc27d` |
| validation224 | `e19fdd7337c260e00cac6678fb6d2249a550c7ec52586bf385959eef43749157` |
| validation320 | `cc5b616bfafa926d21ac0d4faefc71b9ed1043d3baea71f938f2e426cf1fd276` |

Core GPU versions and Docker revision are fixed to the actual verified v13 environment; its full pip-freeze hash is preserved. The new worker verifies core versions and records its full snapshot. No new GPU runtime/model success is claimed. The CPU preparation actually used NumPy2.1.3/Pillow13.0.0.

## Commands and the one remaining training authorization

Use the existing secure recovery environment (no credential values in commands):

```sh
export KAGGLE_CONFIG_DIR=/workspace/shared/trustretina-quota-config
/workspace/shared/trustretina-quota-venv/bin/python scripts/candidate_experiment.py status
/workspace/shared/trustretina-quota-venv/bin/python scripts/candidate_experiment.py check-quota
# Only on a fresh task when actual private inputs are missing:
/workspace/shared/trustretina-quota-venv/bin/python scripts/candidate_recover.py
```

The idempotent recovery command was exercised against existing verified inputs and preserved them without downloads. The fresh-input path selects pinned actual files and verifies bytes/receipts; it was not redundantly redownloaded as a second full copy. No approval is created. The private queue currently has zero GPU attempts and both fits awaiting_approval.

After review/merge and a later explicit exact execution approval, save its bindings to private `artifacts/candidate-v14/experiment/approval.json`. The [disabled template](approval-template.json) lists exact identities/run IDs, cap14,400s/max3 and the required acknowledgement that external confirmation remains blocked. It is intentionally approved=false with no approval time and cannot authorize a push. Then:

```sh
/workspace/shared/trustretina-quota-venv/bin/python scripts/candidate_experiment.py advance --submit --approval-file artifacts/candidate-v14/experiment/approval.json
```

Each command performs one bounded advance. Inspect saved state first and wait at least180s between polls. It never duplicates an ambiguous/prepared job. `reconcile --run-id <already-reserved-run-id> --kernel-version <its-recorded-owner/kernel/1>` validates existing source without a new submission. `retrieve` verifies existing terminal output and resume checkpoints. A current full14,400 free seconds, paid scaling disabled and compatible resources/accounting are mandatory before the first fit; a historical quota observation is not a reservation. No unbounded background monitoring is installed.

## Blocked independently of technical training readiness

BRSET1.0.2 public dictionary/access requirements were rechecked; credentialing and DUA links redirect to login. No restricted metadata/files, account creation, agreement acceptance or access request was attempted. The user must obtain [PhysioNet credentialing](https://physionet.org/settings/credentialing/), complete [CITI Data or Specimens Only Research](https://physionet.org/content/brazilian-ophthalmological/view-required-training/1.0.2/) and personally [sign the BRSET DUA](https://physionet.org/sign-dua/brazilian-ophthalmological/1.0.2/). No fee or restricted-data upload is assumed. [Access evidence](brset-access-readiness.json).

Actual BRSET pair/support/overlap and ODIR-label compatibility remain unverified; no forced mappings or missing-as-negative handling. No full-eight independent cohort is established. These block external evaluation/confirmation and clinical claims. A future TRAIN-only GPU request must explicitly acknowledge this limitation; technical readiness does not silently waive the planning data gates. Calibration fitting, external inference, threshold changes and demo replacement remain outside the training approval. The original demo stays REVIEW REQUIRED.

## Verification and implementation history

Nine new CPU guard/record-consistency tests passed locally; two additional Torch CPU fixture tests cover cached-head alignment/state invariance and strict checkpoint/RNG/scheduler reload. They were skipped locally because Torch is absent and are exercised in CPU CI, whose observed result is recorded separately. Existing v13 checkpoint/training primitives are reused; no new GPU smoke or real model fitting is performed by these tests.

Code review before any GPU execution corrected validation patient-label and quality-ID export alignment, preserved newer attempt state during input restoration, and persisted unexpected provider version responses. Unexecuted intermediate freeze snapshots are retained under history/; no model result was overwritten. The single completed CPU input job and its receipt/source stayed unchanged. Source hashes reflect the final reviewed implementation, not those earlier snapshots.

[Executable contract](../../docs/CANDIDATE_READY_V14.md) · [Preserved prospective design](../../docs/CANDIDATE_FREEZE_V14.md) · [Original dataset review](../../docs/INDEPENDENT_DATASETS_V14.md)

Historical accelerator correction: static metadata from all six v13 checkpoint CUDA RNG lists proves two visible devices, while optimization used only cuda:0. The new guard matches that setup; no DDP or second-device training, no new GPU probe, and the quota cap is unchanged. Actual account counters, not device-count multiplication, determine charged usage. [Evidence](accelerator-evidence.json).
