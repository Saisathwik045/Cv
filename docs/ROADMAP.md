# Roadmap

| Phase / branch | Work | Acceptance criteria |
| --- | --- | --- |
| 0 / phase/00-foundation | Repository, checks, design, runtime and runner audit | Installable package; checks pass; portable validated configuration; honest capability report; runner/account gaps documented; local commit and PR if access permits; no training |
| 1 / phase/01-data | Workbook/image validation and persistent splits | One script validates original data.xlsx, IDs, both-eye references, binary labels, decoding and duplicates; immutable hashes; four patient-disjoint manifests; per-class positive/negative support; fail if infeasible; deterministic rerun and leakage tests |
| 2 / phase/02-baselines | Preprocessing, five models, training and evaluation | All backbones expose eight logits and dropout heads; common protocol; synthetic CPU smoke, interruption/resume equivalence and shape tests; verified runner; five real runs; best/last checkpoints and complete provenance; predictions and metrics persisted |
| 3 / phase/03-reliability | Confidence, MC Dropout, quality, calibration, triage | BatchNorm frozen during stochastic dropout; calibration only on reserved data; thresholds frozen before evaluation; all ablations from same examples; risk/coverage and calibration results including failures; no forced improvement |
| 4 / phase/04-explanations-app | Grad-CAM, inference and upload app | Versioned inference bundle reproduces preprocessing/thresholds; class-specific maps and sanity checks; safe upload decoding/limits; ACCEPT or REVIEW REQUIRED reasons; no storage by default; automated end-to-end test |
| 5 / phase/05-report | Locked final evaluation and report | Final evaluation opened once for frozen pipeline; patient bootstrap intervals; comparisons and figures regenerated solely from real artifacts; missing runs explicit; inference-only official test if unlabeled |

Each phase uses a separate PR from current main. A phase whose external execution
is blocked records that blocker rather than manufacturing completion. Later GPU
training must never be triggered by the lightweight PR workflow.
