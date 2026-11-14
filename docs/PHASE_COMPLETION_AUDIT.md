# Phase completion audit through V15

Audited from public committed sources only after merged PR #32. No private patient data or historical-final cohort was opened. Complete means the recorded development stage completed; it does not mean clinical validation.

| Phase/version | Stage | Audit status | Evidence and boundary |
|---|---|---|---|
| 0 | Foundation | Complete | Package, configuration, declarative planning and capability checks. [Source](PHASE0_REVIEW.md) |
| 1 | Data validation and initial splits | Complete; early split acceptance superseded | Workbook/image checks implemented; later review revised the experimental release. [Source](DATA_PIPELINE.md) |
| 2 | CPU research modules | Complete | Five factories, preprocessing, evaluation, checkpoint and authorization contracts. [Source](PHASE2_CPU.md) |
| 3 | Bounded duplicate review | Completed through later v4 resolution | Initial partial review correctly blocked execution; historical weak-hash grouping was not accepted as truth. [Source](DUPLICATE_REVIEW_V2.md) |
| 4 | Contained release and execution pilot | Complete | Fixed scheduled review and explicit containment; one bounded random-initialization pilot, not performance evidence. [Source](DEVELOPMENT_V4.md) |
| 5 | Five-backbone comparison | Complete | All five ImageNet-initialized development fits and aggregate comparisons. [Source](../results/full-v5/README.md) |
| 6 | Original reliability/calibration/explanations | Complete; clinical policy unsupported | One original shared temperature, head-only MC, technical quality and Grad-CAM; REVIEW REQUIRED. [Source](../results/reliability-v6/README.md) |
| 7 | Historical frozen evaluation | Recorded complete; not reopened | Status records a once-run frozen evaluation; no holdout records or replay used for this submission. [Source](FINAL_EVALUATION_V7.md) |
| 8 | Streamlit research demo and Mac launcher | Implemented; native Mac verification outstanding | Original frozen candidate only; local CPU/synthetic UI and portable launcher checks. [Source](DEMO_V8.md) |
| 9 | Controlled development runs | Complete | Four fixed schedule/backbone runs; reused-validation interpretation. [Source](../results/round2-v9/RESULTS.md) |
| 10 | Exploratory threshold CV | Complete; thresholds not deployed | Group-aware threshold assessment does not undo prior checkpoint-selection exposure. [Source](../results/threshold-v10/RESULTS.md) |
| 11 | Nested training-only loss comparison | Complete | Six paired fits; square-root weighting did not establish superiority. [Source](../results/nested-v11/DISCUSSION.md) |
| 12 | CPU development audit | Complete | Saved evidence reviewed; hypotheses about H/O/quality are not proven causes or clinical relabeling. [Source](../results/development-audit-v12/REPORT.md) |
| 13 | Matched-resolution study | Complete; independent confirmation absent | Six fits; conditional development evidence nominated320 with class-specific harms retained. [Source](../results/resolution-v13/DISCUSSION.md) |
| 14 | Two full-role frozen candidates | Complete | Matched320/224 fits, strict reload and saved post-freeze outputs; original demo preserved. [Source](../results/candidate-v14/RESULTS.md) |
| 15 | Real CPU calibration and reliability | Complete; development only | Two scalar temperatures and all prespecified reliability ablations; negative ECE/quality findings published in merged PR32. [Source](../results/calibration-v15/RESULTS.md) |

## Current boundaries

- Development implementation and real analysis through V15: complete in the public record.
- Independent all-eight confirmation and clinical policy: incomplete/unsupported.
- Original demo and later V14/V15 candidates: distinct, unchanged.
- Native macOS execution: unverified; portable/Linux checks are not native evidence.
- Durable encrypted backup/restore drill: outstanding, no transfer performed.
- Institution-specific report formatting, author details and supervisor approval: author actions.

See the [final report](FINAL_RESEARCH_REPORT.md), [presentation outline](FINAL_PRESENTATION_OUTLINE.md) and [submission checklist](FINAL_SUBMISSION_CHECKLIST.md).
