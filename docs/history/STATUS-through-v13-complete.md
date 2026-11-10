> Archived v13 closeout before PR25 merged. Current state is in ../STATUS.md.

# V13 matched resolution study — completed

All six authorized training-only fits completed and were retrieved/source/nonce/output-hash verified. The complete paired analysis and 2,000 fixed-seed component bootstrap replicates are finished. [Results](../results/resolution-v13/RESULTS.md), [all-eight findings and plots](../results/resolution-v13/DISCUSSION.md), [registry](../results/resolution-v13/queue.json), and [accounting/artifact hashes](../results/resolution-v13/execution-provenance.json).

| Run | Status | Epochs / best (one-based) |
|---|---|---:|
| fold0_res224 | verified complete | 13 / 7 |
| fold0_res320 | verified complete | 11 / 5 |
| fold1_res224 | verified complete | 13 / 7 |
| fold1_res320 | verified complete | 15 / 9 |
| fold2_res224 | verified complete | 10 / 4 |
| fold2_res320 | verified complete | 12 / 6 |

Settled free Kaggle charge: **6926.895s (1.924138h)** against 21,600s. Exactly 6 submitted sessions; zero failures/resubmitted GPU sessions, zero final reserved time, paid scaling disabled. Worker elapsed time is separately 6717.541s. Owned remote inventory matches the six registered version1 jobs. No seventh session, additional training or refit is authorized.

| Mean outer metric | 224 | 320 |
|---|---:|---:|
| Macro AUROC | 0.83273 | 0.85108 |
| Macro AP | 0.57279 | 0.59412 |
| Macro F1 | 0.53758 | 0.57156 |
| Micro F1 | 0.53534 | 0.57509 |

Primary paired macro AUROC difference (320−224): **+0.01835**, conditional 95% interval **[+0.01036,+0.02655]**. Macro AP and F1 differences are positive on this cohort. The frozen numerical rule nominates 320 for future research only; no fold checkpoint is promoted globally. These intervals condition on fitted models and omit training/selection variability; prior adaptive same-source development and one seed limit independence. Fresh compatible independent paired-eye eight-label data is needed for confirmation.

H pooled recall0.1791→0.2537, FP25→76, FN55→50, F10.2308→0.2125 (67 positives/66 positive components). Its paired recall/F1 intervals include zero. O recall0.4450→0.6588, FP439→720, FN353→217, F10.4168→0.4721 (636 positives). G AP/F1 are worse; D/A/M have more missed positives at p≥.5. Keep these harms alongside aggregate gains. H/O source semantics and patient-derived per-eye targets remain limitations, without relabeling or clinical inference.

Frozen approval identities remain unchanged:

- Plan `0fa597609d0390e7414865ed9507cd97e214c99945abd4c1cc095a9924b1cf2a`
- Worker/source `e8a8df1bbca95b14109f4e16ca31080e9dd9afe4746d42af5e228b96c0a5e80c`
- Configuration `fcc8c73b752551a5d5b860b3d134e2828b78ea7a4a5aa41c48549b3a717ae0b4`

No image preparation, split generation, duplicate review, GPU smoke or historical final evaluation was repeated. All five factories, both loss implementations, frozen demo, calibration, thresholds and REVIEW REQUIRED behavior remain unchanged.

CPU checks: two focused execution-guard/archive-verification tests passed; syntax/diff checks and CPU dependency consistency passed. All six real output audits replayed fit-only weights, inner-only selection, strict reload evidence, frozen outer alignment and mean-eye prediction metrics. The bootstrap resumed from 1,580 saved replicates after its local process disappeared, using identical seeds/sources; no model rerun or analysis change. The frozen protocol, worker, controller and reporter bytes are unchanged. Implementation/results commit `c6a3932`: all seven required CI jobs passed in [run42776144015](https://github.com/Saisathwik045/Cv/actions/runs/42776144015). This final status-only update preserves those implementation bytes.

PR: https://github.com/Saisathwik045/Cv/pull/25 (focused execution/results branch; not merged).

Next executable action is CPU-only artifact recovery or independent review using [RECOVERY.md](../results/resolution-v13/RECOVERY.md). No GPU job is queued and no monitoring is promised after closeout. Further training, refit, threshold/calibration work or demo replacement requires a separate scoped decision. Private exact receipts and TASK_RESUME.json preserve recovery; historical preparation is preserved [here](../results/resolution-v13/README.md) and [prior status](history/STATUS-through-v13-preparation.md).
