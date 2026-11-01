# Phase 9 — all four approved experiments complete

PR #18 was already merged at `26f4a51`; final CI `41283819937` passed. The exact approved plan/source/configuration were reused. No new smoke, dataset preparation, duplicate review or split generation occurred.

| Run | Epochs / best | Patient macro AUROC | Macro F1 | Micro F1 | FP / FN label decisions |
|---|---:|---:|---:|---:|---:|
| original_convnext_tiny | 5 / 2 | 0.8288 | 0.4762 | 0.4988 | 681 / 189 |
| original_densenet121 | 6 / 3 | 0.8119 | 0.4607 | 0.4773 | 753 / 191 |
| convnext_constant | 13 / 7 | 0.8391 | 0.5558 | 0.5455 | 322 / 268 |
| convnext_cosine | 19 / 13 | 0.8417 | 0.5926 | 0.5714 | 243 / 276 |
| densenet_constant | 10 / 4 | 0.8142 | 0.4900 | 0.5027 | 647 / 196 |
| densenet_cosine | 10 / 4 | 0.8209 | 0.4781 | 0.5003 | 589 / 218 |

**Selected development run: ConvNeXt cosine, epoch 13**, by prespecified maximum patient validation macro AUROC. All use the same 525 validation patients / 518 containment components, mean-eye sigmoid aggregation and fixed p ≥ .5. [Generated report](../../results/round2-v9/RESULTS.md) includes all-eight AUROC/AP/support/precision/recall/F1 and FP/FN, accuracy definitions, original baselines and learning curves.

The selected run reduces FP label decisions 681 → 243 but increases FN 189 → 276. N and D F1 fall; H recall falls 11/16 → 3/16 despite higher H F1. DenseNet cosine improves ranking over its constant counterpart but lowers macro F1. These negative findings are retained. V9-versus-v5 also changes checkpoint/early-stop/budget rules; only the within-v9 constant/cosine contrasts isolate the schedule.

Actual Kaggle quota counter increase: **4815.738 seconds (1.337705 hours)** across five sessions including the one ConvNeXt resume. Worker time 4647.982 seconds. Conservative charged reservations: 12,000 seconds (3h20m), below the four-hour cap. No paid scaling; final API shows zero reserved GPU time. The bounded driver returned; no training or monitoring continues after this task.

## Verified artifacts and recovery

All best/last checkpoints, histories, logs, image/patient predictions, exact source versions and receipts were retrieved into ignored private storage and checksum/source/nonce verified. Strict best reload passed. The ConvNeXt cosine resume preserved epochs 1–16, continued at 17, and matched the saved next LR; final history has 19 epochs. Patient probabilities were independently reconstructed from both eye predictions; labels/IDs/components match original validation artifacts exactly. A fresh-directory download of the first run’s predictions/history/report/checksum also reproduced the saved hashes without submitting a job.

[Public registry and exact pinned versions](../../results/round2-v9/queue.json) · [aggregate execution/hash provenance](../../results/round2-v9/execution-provenance.json) · [read-only recovery commands](../ROUND2_RECOVERY.md). Private authoritative registry: `artifacts/runner/round2-v9/registry.json`; approval, source/submission/verification receipts and outputs remain there. Private Kaggle versions retain every completed and paused attempt; never resubmit them.

Frozen demo checkpoint, candidate contract, temperature, normalizer, pipeline, calibration/final manifests and historical final report/predictions remain byte-identical (10 opaque SHA256 checks). No historical final or calibration arrays were parsed for this experiment. All five original backbone implementations remain unchanged; REVIEW REQUIRED remains the deployment default. No newly trained checkpoint is installed in the demo.

## Checks and next step

CPU scientific environment: 188 tests, 9 optional skips (179 passed). Foundation environment: 188 tests, 29 optional skips (159 passed). Compilation, configuration checks and `git diff --check` pass. Remote CI for this results PR is tracked separately; local tests are not remote training evidence.

The reused validation cohort can support this fixed development comparison, with repeated-selection optimism, one seed and H support of 16. It cannot establish a new generalization claim. Next: review these results; define a fresh suitable independent evaluation cohort before claiming improved performance. Any threshold experiment must be separately prespecified on development data, with calibration fitting separate. No further training or fitting is queued.

[Original prospective plan/status](STATUS-through-phase9-plan.md) · [frozen protocol](../CONTROLLED_TRAINING_V9.md). The protocol’s original approval-pending wording is preserved because its hash is bound to the separately recorded execution approval.
