# Final academic presentation outline

Suggested duration: 15–20 minutes plus questions. Use the
[final report](FINAL_RESEARCH_REPORT.md), [phase audit](PHASE_COMPLETION_AUDIT.md)
and existing public aggregate figures. Author/institution/supervisor details and
institutional slide formatting remain student-supplied. Independent validation is
incomplete; every research decision remains **REVIEW REQUIRED**.

| Slide | Message | Evidence / speaker guidance |
|---|---|---|
| 1. Title/scope | TrustRetina: reproducible paired-eye retinal screening research | Development study, not clinical certification; identify author and institution. |
| 2. Problem | Multi-label estimates need honest reliability and leakage controls | Explain why a single accuracy number is insufficient. |
| 3. Objectives | Compare fixed backbones and assess calibration, uncertainty and quality | Separate engineering completion from independent confirmation. |
| 4. Data/ethics | ODIR patient-derived N,D,G,C,A,H,M,O | Explain patient/eye inheritance, H/O semantics, unknown labels and privacy. |
| 5. Architecture | Preprocessing → eight logits → mean-eye probabilities → calibration/reliability → REVIEW REQUIRED → optional Grad-CAM | Use the report's pipeline diagram. |
| 6. Roles/leakage | TRAIN2,275, validation525, calibration350; components remain whole | Superseded weak-hash grouping and later explicit containment; no patient IDs. |
| 7. Models/training | Five fixed ImageNet-initialized backbones | Distinguish v5 validation-loss checkpoints from later AUROC selection. |
| 8. Original comparison | ConvNeXt-Tiny selected on development macro AUROC | Show all five rows; disclose selection optimism and sparse H. |
| 9. V9–v13 | Controlled schedule, threshold, loss, audit and resolution studies | Keep increased FP, weak loss evidence and conditional resolution findings. |
| 10. V14 | Two distinct frozen320/224 candidates,11 epochs/best5 | Show class tradeoffs; no uniform superiority or demo replacement. |
| 11. V15 method | One component-equal scalar per checkpoint on350 patients/331 components | Post-mean-eye sigmoid calibration, bounded optimizer and fixed0.5 decisions. |
| 12. V15 results | T320=1.105496125613; T224=1.002532561130 | Separate in-sample calibration from reused-validation NLL/Brier/ECE. |
| 13. Negative findings | ECE worsened; classification stayed unchanged | Use all-class reliability panels, not only favorable classes. |
| 14. MC/quality |20 trained-head passes; technical quality fusion worsened AURC | Paired whole-component intervals include zero between resolutions. |
| 15. Demonstration | Original Phase6 demo differs from V14/V15 results | Synthetic UI/fake-service tests only in this task; mark synthetic content explicitly. |
| 16. Reproducibility | Source hashes, version locks, guards, tests and CI | A digest/software test is not independent clinical validation. |
| 17. Limitations | In-sample calibration, reused validation, sparse support, label semantics | Include external access, native Mac and private-backup gaps. |
| 18. Conclusion | Development work complete; independent confirmation incomplete | REVIEW REQUIRED and prospectively specified next study. |

## Demonstration run sheet

1. Use the [submission checklist](FINAL_SUBMISSION_CHECKLIST.md) synthetic tests.
   Generated PNGs/fake outputs need no private weights, data or credentials.
2. Show paired-input blocking, invalid-image handling and visible missing-bundle
   failure. No random model fallback is acceptable.
3. Show eight-class presentation, REVIEW REQUIRED, explanation-target selection,
   download-control presence and reset. Identify the service and values as fake;
   these are software checks, not real predictions or clinical evidence.
4. Use published aggregate V15 diagrams for science. Any real frozen-model/image
   demonstration requires separately permitted bundle/image access; none is run here.
5. Never put private cases, original manifests, patient-level arrays or credentials
   into slides, screenshots, recordings or submission assets.

## Viva questions and accurate answers

Use the [research defence chapter](ACADEMIC_RESEARCH_DEFENCE.md) as an examiner
appendix and the [all-eight-class table](ALL_EIGHT_CLASS_COMPARISON.md) for the
class-harm slide. The **Research study & viva** dashboard page shows the questions,
recorded V15 metrics and reviewed reliability figures without running models.

Suggested appendix slides: research question→evidence→limitation matrix;
experiment/ablation map with selection exposure; all-eight-class FP/FN and ECE;
observation versus causal hypothesis; contribution/assistance disclosure;
reproduction levels and native-Mac/cloud-lifetime limitations. Complete author
contribution details truthfully before submission. These additions do not extend
the scientific claims or authorize private inference.

- Why no independent validation? Validation informed selection and calibration is
  in sample; a compatible untouched external cohort has not been established.
- Why can NLL improve while ECE worsens? The scalar optimizes component-equal NLL,
  not bin-based ECE or each class separately.
- Why unchanged FP/FN/F1? Positive shared temperature preserves the0.5 decisions;
  execution verified no numerical/clipping crossing. It does not fix discrimination.
- Does MC variance establish safety? No; head-only variability omits backbone
  uncertainty and is not a clinical confidence interval.
- Why keep the old demo? Its frozen checkpoint/temperature/normalizer contract is
  separate; V14/V15 do not authorize replacing or clinically promoting it.
- Does Grad-CAM prove lesions? No; target sensitivity on a transformed canvas is
  not lesion-ground-truth validation or proof of prediction correctness.
- What is complete? The implementation and development study through V15;
  independent confirmation, clinical policy, native Mac verification and durable
  private preservation remain separate gaps.
