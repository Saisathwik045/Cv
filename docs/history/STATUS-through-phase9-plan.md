# Phase 9 — controlled second training round prepared; approval required

Continue from latest merged main `d7d7187` (Mac packaging PR #17 merged).
[Prior demo/Mac status](STATUS-through-phase8-macos.md) preserves evidence
and native-Mac limitations. The demo checkpoint, contract, five backbone factories
and REVIEW REQUIRED behavior are unchanged. No new training job was submitted.

## Verified diagnosis

All five v5 job versions remotely report COMPLETE. Original source/nonce/report,
best/last, history, predictions and shared train/validation cache hashes verify;
selected-checkpoint patient metrics reproduce from private validation arrays.
ConvNeXt remains the historical winner at validation macro AUROC 0.828764.
Historical held-out artifacts are preserved separately and are not tuning inputs;
no holdout prediction arrays or individual labels/errors were loaded.

[Generated development diagnosis](../../results/round2-v9/DIAGNOSIS.md) and
[audit](../../results/round2-v9/development-audit.json) show falling training loss
with worsening validation loss for ConvNeXt/DenseNet: evidence of overfitting or
overconfidence, not proof that more epochs help. Loss-selected checkpoints differ
from their best recorded AUROC epochs (ConvNeXt epoch 2 vs 4; DenseNet 3 vs 4).
Rare-label positive weighting and fixed .5 decisions contribute high FP counts:
ConvNeXt H predicts 183 positive among 525 validation patients, with 16 positives
(11 TP / 172 FP / 5 FN). N/D/O ranking is also limited. No labels were altered.

Frozen v4 stays 2,275 training / 525 validation patients, 2,240 / 518 components,
both eyes together and zero train/validation patient/component overlap. Root and
train/validation manifest hashes match. All-eight validation positive counts:
N,D,G,C,A,H,M,O = 175,169,32,32,25,16,26,147. Full negative/supplying-group support
is in the audit. Reused validation is suitable only for this bounded exploratory
comparison; it is not fresh evaluation and H estimates are unstable.

## Prospective matrix and boundary

[Protocol](../CONTROLLED_TRAINING_V9.md) and
[configuration](../../configs/controlled-training-v9.toml) define four new fits:
ConvNeXt-Tiny constant / cosine LR, then DenseNet121 constant / cosine LR.
All use ImageNet V1 from the existing verified cache, seed 2026, FP32 AdamW
1e-4 / weight decay .01, batch 16, 224 pixels, no augmentation, unchanged weighted
BCE and mean-eye sigmoid aggregation. Cosine decays to 1e-6 over 25 epochs.
Maximum 25 epochs, six-epoch early stopping on validation patient macro AUROC
(delta 1e-4); strict best-AUROC checkpoint, earliest exact tie. All four completed
runs are compared by that same primary criterion, fixed .5 secondary metrics.

This establishes a new AUROC-selected baseline; only LR schedule changes within
each v9 pair. Historical v5 loss selection is preserved. No seed search.
Higher resolution, alternative imbalance loss, OOF threshold fitting, calibration
and risk fitting remain outside this queue and require separate design/budget.
The future threshold objective/grid/tie/support rule is prospective and tested
synthetically; no thresholds were fitted on saved real data.

Measured-worker extrapolation for all four at 25 epochs: **2.376 h**. Request
**four GPU-quota hours**: maximum six half-hour job attempts (three worker hours),
each charged a ten-minute setup/teardown guard; max two attempts per run. Failed
and ambiguous attempts count. Interruptions can exhaust the cap before all four
finish; no automatic extra run or paid fallback. Current read-only quota reports
six free hours, zero used/reserved and paid scaling false; recheck before jobs.
API accounting/provisioning can lag and is not a second-exact billing guarantee.

## Implementation, recovery and next action

New isolated v9 worker/controller retain scheduler/model/optimizer/all RNG state,
per-epoch LR/all-eight validation metrics, best/last checkpoints and checksums.
ImageNet initialization and original input/protocol/source/runtime hashes are bound
in provenance. Private pinned Docker/runtime is reused. Existing IDs/receipts and
remote inventory must be reconciled before any submission; running/completed/
ambiguous jobs cannot be duplicated. Private attempt outputs are durable on Kaggle.
No calibration/final data source is mounted. No historical final rerun is allowed.

CPU preparation: `python scripts/controlled_experiment.py prepare`; `status`,
`check-quota` and `restore` are read-only/recovery operations. The concrete
[queue snapshot](../../results/round2-v9/queue.json) records source/config/plan hashes
and all four pending runs; private registry/index: `artifacts/runner/round2-v9/`.
Preparation does not submit jobs. `advance --submit` rejects missing or mismatched
explicit user approval before creating an API client. After approval, bind its
reference to the exact plan/source/run IDs/four-hour cap in private JSON and
advance one sequential job at a time. Preserve incomplete/unsuccessful results.

Local full suite: 185 tests run, 176 passed and nine optional skips; all 16
focused v9 tests passed. Compile and diff checks passed. Tests cover AUROC
ties/patience, exact matrix/role guards, free-
quota/budget caps, no approval/no submission, ambiguous/unknown/completed/active
job handling, write-ahead receipts, private-source bundles and exact synthetic
scheduler/checkpoint resume. These are not new remote GPU execution evidence.
A new generalization claim requires a fresh suitable independent evaluation
cohort and prospective protocol, with patient/identity containment and adequate
all-eight label support. The old exposed holdout cannot select this round.

**Single remaining approval:** execute the exact four-run matrix with a maximum
four-hour free-Kaggle GPU-quota budget and the declared reservation/attempt caps.
