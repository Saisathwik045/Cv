> Historical preparation record below. The user authorized this exact plan on9November2026; execution state and results are tracked in [STATUS](../../docs/STATUS.md) and [queue](queue.json). Pending-approval statements below describe preparation only.

# Matched resolution study v13 — prepared; training not authorized

[Frozen protocol](../../docs/RESOLUTION_V13.md) · [configuration](../../configs/resolution-v13.toml) · [prerequisites](prerequisites.json) · [identities](frozen-identities.json) · [queue](queue.json) · [all-eight fold support](support.csv)

## Verified prerequisites

PR #23 is already merged at `d94fbda`; CI41311898604 passed all seven jobs. Current preparation is on latest merged main plus this focused branch. No prior CI-unavailable result was treated as a pass.

One private CPU job `saisathwikregula/trustretina-res-v13-inputs-767673ae/1` completed (GPU disabled, 763.3s worker time). It recovered **4,550 original TRAIN images / 2,275 patients**, checked every frozen raw SHA256, reproduced224 from native decode and found **zero differing uint8 channel values on all4,550 images**, then built320 directly from originals. Retrieved report/source/nonce and every output hash passed; independent local streaming rehash verified all4,550 archived originals without extracting unrelated patients. The TRAIN-only archive, per-image receipts and both cache tensors are retained privately in that exact Kaggle output version; no new data source or original-image upload was needed.

Provider metadata normalizes the requested `/2` dataset source to its unversioned slug. Consequently provenance is pinned by the observed version2 source plus **all frozen raw-byte hashes**, not by claiming the returned metadata preserves a version suffix. The raw hash guard would block changed content. Direct original download was blocked at a Google Storage redirect; authorized Kaggle CPU preparation avoided needing that host, without changing network policy.

Pinned preparation dependencies: Pillow13.0.0, NumPy2.1.3; original training Docker/dependency contract remains pinned. The224 tensor SHA256 is `274679427634724590dd9a7879cd562879e3d49dea6fdf62aa4e84a7d51fc70f`;320 is `9cd180cc9fff089b87320ad1a6f097c8b632a277e41dee5b54e06c9ecb0bc27d`. Full source/index/raw-archive/initialization identities are in prerequisites.json.

Authoritative [ODIR dataset](https://odir2019.grand-challenge.org/dataset/) and [creators repository](https://github.com/nkicsl/OIA-ODIR) confirm patient-level H=hypertension and O=other diseases/abnormalities. Precise H case criteria, exhaustive O inclusion rules and eye-localized truth remain unspecified. This permits a controlled study of unchanged dataset targets, not clinical hypertension/lesion claims or new-cohort harmonization. See [source findings and access limitations](source-definitions.json). No relabeling.

## Exact matched matrix

| Order | Run ID | Resolution | Worker timeout | Quota reservation incl. guard | State |
|---|---|---:|---:|---:|---|
| 1 | fold0_res224 | 224 | 2400s | 2700s | queued; approval required |
| 2 | fold0_res320 | 320 | 4200s | 4500s | queued; approval required |
| 3 | fold1_res224 | 224 | 2400s | 2700s | queued; approval required |
| 4 | fold1_res320 | 320 | 4200s | 4500s | queued; approval required |
| 5 | fold2_res224 | 224 | 2400s | 2700s | queued; approval required |
| 6 | fold2_res320 | 320 | 4200s | 4500s | queued; approval required |

Only resolution changes between arms. Fresh official ImageNet ConvNeXt-Tiny in every fit; common224/320 preprocessing geometry/RGB/channel normalization, ratio BCE from fit patients only, seed2026, FP32 AdamW/cosine1e-4→1e-6, weight decay.01, no augmentation, max25 epochs. Common microbatch8 × accumulation2 = effective16; exact sample weighting for the partial last window. Best checkpoint maximizes inner patient macro AUROC, earliest tie; six-epoch meaningful-improvement patience with min delta1e-4. Save best/last and all resume state every full epoch. Outer assessment is deferred until complete training and strict best reload.

V11 reference predictions/checkpoints cannot substitute for the new224 arm: microbatch16 without accumulation differs from8×2 in stochastic-mask/kernel/numerical execution. Both arms must be rerun under the same implementation; old fits are historical execution/cost evidence only. No continuation of v11/v9/demo weights. All five factories and both historical loss implementations are unchanged.

Mean-eye sigmoid probabilities at p≥.5 remain fixed. Primary contrast: mean paired outer macro AUROC, with fixed2000 seed2026 within-fold paired component bootstraps. Report macro AP/macro+micro F1 and all-eight AUROC/AP/precision/recall/F1/TP/FP/FN/support, especially H. No improvement claim from an interval including zero, no hiding worse F1/rare-class results, no partial-matrix winner, no threshold tuning/calibration/full-data refit. Earlier adaptive decisions still limit independence.

Frozen v11 fit/inner/outer patient counts remain1137/379/759,1137/380/758,1137/380/758. H positives remain33/11/23,33/12/22,34/11/22. All-eight per-role support and component counts are unchanged in support.csv; support floors are not power guarantees.

## Budget and actual quota blocker

Area-scaled all25-epoch estimate: **4.23 worker-hours**, or **5.50 with30% allowance**. This is not a measured new GPU throughput or memory guarantee. Propose **21,600 free GPU-quota seconds (six hours), maximum six sessions**, including all failures/resumes/overhead. Reservations total exactly21,600 seconds; no seventh session or paid fallback.

Observed free quota: **12670.775s (3.5197h)** at `2026-11-01T15:28:46.139875+00:00`; reserved0, paid scaling disabled. CPU preparation changed GPU usage by **0.000s**. The full-six-hour first-submission gate currently fails. Provider reports refresh at **7 November2026,05:30 IST**; this is a reported refresh, not guaranteed future availability. Recheck actual free quota before starting.

SDK exposes used/reserved/total time and paid flag, not a guaranteed per-accelerator billing multiplier or live T4 availability. Prior T4 evidence:3953.290 worker seconds versus4113.487 charged quota seconds. New requested accelerator is NvidiaTeslaT4; use actual settled quota-counter deltas, separate overhead and conservative observed-rate guards. Stop safely if accounting or runtime cannot cover the next reserved session. Current lack of full quota is an execution blocker, not grounds to lower epoch budgets or assume early stopping. No GPU probe was run.

## Frozen approval target

- `plan_sha256`: `0fa597609d0390e7414865ed9507cd97e214c99945abd4c1cc095a9924b1cf2a`
- `experiment_source_sha256`: `e8a8df1bbca95b14109f4e16ca31080e9dd9afe4746d42af5e228b96c0a5e80c`
- `config_sha256`: `fcc8c73b752551a5d5b860b3d134e2828b78ea7a4a5aa41c48549b3a717ae0b4`
- `protocol_sha256`: `04bb5d9885bea00f3a0148310d1340db9fc6286a7e497682d2a6e4b9c8119775`
- `controller_sha256`: `ba8edfc73b037941055b825f7bf369fe817359b15ffda0f6b96306ff3d49641a`
- `reporter_sha256`: `fb21c730fcef0e1b250ecaff8976cd72dbcee29a499ea31f2add5f7a68c06016`
- `source_commit`: `781e4a9d1a17743bf0a7e9adbf6596711dd8667c`
- `fold_manifest_sha256`: `e8f3e7a6e44880c2a5d09dac08619df029d650658d720b26b15352721ca844a5`

Approval must bind this exact plan/source, six listed run IDs,21,600 free-quota seconds and maximum six sessions. No approval record has been created. Training may start only after separate user approval, reviewed checks and observed full quota availability.

## Checks and recovery

225 CPU tests:216 passed, nine optional skips;11 focused tests cover no-approval/no-API guards, fixed settings, failure/resume budget limits and charging, exact accumulation including partial batches, stochastic checkpoint resume, resolution provenance, native-only archive hashes, source-resolution detail and conflicting-state recovery. Compilation and diff checks pass. Actual official-ImageNet CPU forwards on synthetic zero tensors are finite (2,8) at224/320; no patient inference/training or GPU memory test. See cpu-forward-check.json.

Private state: `artifacts/resolution-v13/TASK_RESUME.json`, preparation receipt/source/downloads and experiment registry/fold manifest. Cache/raw archive live in the existing pinned private CPU job output, not Git. On a fresh authorized task, `python scripts/restore_resolution_state.py` restores pinned outputs/receipts and the exact frozen fold manifest from the verified original v11 spec; it does not create new splits or submit jobs. It refuses conflicting/newer local state and creates no approval record. Set up the existing secure local Kaggle SDK credentials without embedding them.

`python scripts/resolution_experiment.py status` reads the local queue; `check-quota` is read-only. Future `advance --submit --approval-file <private-exact-approval.json>` is approval-gated, checks existing provider jobs/receipts and quota, and never blindly repeats ambiguous/prepared saves. `python scripts/report_resolution_results.py --bootstrap-batch 100` is CPU-only and refuses an incomplete matrix. There is no background monitoring. The CPU preparation job is complete; **zero training/GPU sessions submitted**.

Demo/temperature/normalizer/contract hashes match the frozen bundle; historical evaluation, calibration, thresholds, labels and architecture remain unchanged. REVIEW REQUIRED remains default. New external confirmation data is still required for any generalization claim.
