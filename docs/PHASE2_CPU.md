# Phase 2 CPU implementation — fitting disabled

The original pipeline and five backbones are preserved: image → shared
preprocessing → eight independent logits → sigmoid confidence + MC Dropout
uncertainty + technical quality → calibration → ACCEPT / REVIEW REQUIRED → later
Grad-CAM. Factories are EfficientNet-B0, ResNet50, DenseNet121, ConvNeXt-Tiny and
EfficientNet-B2, in that fixed tie-break order. The present work implements CPU
building blocks, not an executed experiment or the later Grad-CAM/application phase.

Install the tested CPU environment and run checks:

```sh
python -m pip install -r requirements/phase2-cpu.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m unittest discover -s tests -v
python -m compileall -q src tests
trustretina check-config --config configs/phase0.toml
trustretina check-config --config configs/phase1.toml
```

The Linux CPython 3.11/3.12 lock pins direct and transitive runtime dependencies,
including CPU Torch/Torchvision. It is version-pinned, not a hash-locked wheel
supply-chain record. CI has a dedicated locked CPU job in addition to the lightweight
foundation job. No timm is needed for these torchvision factories. No factory
fetches pretrained weights: default random initialization is labeled unfitted;
local ImageNet states require a trusted checksum. A verified weight source/license
and matching state dictionary must be recorded before future experiments.

## Implemented contracts

- Shared EXIF orientation, RGB conversion, aspect-preserving 224×224 letterbox,
  declared ImageNet mean/std, configuration fingerprint and geometry metadata.
  Evaluation has no augmentation. Optional training horizontal flip requires an
  explicit RNG; the read-only dataset currently uses the deterministic baseline.
- Common dropout .2 + eight-logit head, BCEWithLogitsLoss and a gated future
  training loop, validation-only checkpoint/backbone selection, fixed epoch budget,
  last/best checkpoint saves and per-epoch provenance. No training CLI is exposed.
- Atomic epoch-boundary checkpoint serialization verifies a trusted checksum and
  exact backbone/split/config/preprocessing/seed/data-kind provenance; restores
  model/optimizer/scheduler/scaler plus Python/NumPy/Torch/loader RNG state.
  Mid-epoch resume and equivalence across devices/library changes are not claimed.
- A read-only, hash-verified phase2 eye-row adapter enforces both eyes, consistent
  patient labels/groups and manifest role. Historical manifests cannot be consumed
  implicitly: a separately reviewed conversion/release is required. Whole-bundle
  separation checks remain a data-release responsibility; fitting additionally
  binds batches to approved role-specific manifest digests.
- Inference-only evaluation restores module modes and calculates equal-shaped
  two-dimensional (n,8) multilabel F1, precision, recall, per-class AUROC/AP, support
  and Brier scores. Ranking metrics are undefined when both outcomes are absent;
  undefined values are explicit, not silently favorable numbers.
- Primary unit is patient: mean the two eye sigmoid probabilities **before**
  thresholding, with the same original patient-derived label vector. Per-eye results
  are secondary and cannot be described as verified eye labels. Confidence intervals
  resample duplicate-connected patient components, never independent eyes. The
  included macro-F1 component bootstrap is descriptive; rare-class intervals and
  simultaneous claims need additional prespecified analysis.
- Identity/shared/per-class temperature application, support checks and an explicitly
  gated future optimizer interface. No calibrator was fitted. The caller must pass
  patient-level logits derived from the fixed aggregation, not duplicated eye rows.
- MC Dropout enables dropout layers only, leaves batch normalization frozen and
  restores all mode flags. Quality outputs are raw foreground/contrast/gradient
  proxies, not clinically validated gradability. There is no fitted quality normalizer.
  Reliability thresholds/support flags are configurable; default policy always
  returns REVIEW REQUIRED. Risk-policy fitting fails explicitly until the risk loss,
  budget, search family, multiplicity and confidence bounds are reviewed.

CPU tests cover all five untrained forward paths, multilabel versus argmax behavior,
patient aggregation, immutable image/manifest checks, fitting/final-access denial,
checkpoint/RNG serialization without an optimizer step and dropout versus batchnorm
state isolation. Synthetic forward tests establish software contracts only.
They do not validate real-data performance or trained resume equivalence.

## Protocol v2 required before any fitting

This is a prospective specification; `training_ready` remains false. The data review
and revised manifest version must be approved before it becomes executable.

1. Keep both eyes and all confirmed/probable-copy connected patients together.
   Resolve material weak-edge uncertainty or explicitly review a quarantine cohort
   with selection-bias limitations. Retain every raw image/label and prior manifest.
   Publish all-eight-class positive/negative **patient and independent-component**
   support, component sizes, assignment algorithm/seed and immutable digests. Do not
   search seeds for attractive results. Any revised deterministic assignment is
   justified by evidence/protocol changes, never observed model performance.
2. Use validation only to select checkpoints and the backbone: macro F1 on patient
   mean probabilities at fixed .5, ties earliest epoch then fixed backbone order.
   Keep the common 30-epoch, optimizer/schedule/preprocessing budget fixed across
   five backbones. No final or calibration results choose a backbone. Any later
   validation-fitted disease thresholds are a separately declared step after model
   selection, frozen before calibration/final evaluation; default thresholds are .5.
3. After selection, fit one predeclared shared temperature to **patient-level**
   calibration-fit predictions (logit of clipped mean eye probabilities, clipping
   1e-6). Do not compare calibrators on final evaluation. Class-specific temperatures
   are disabled unless a prospective class/component support and power assessment
   supports them. A minimum 20 positive and 20 negative independent patient groups
   per class is an operational refusal floor for each fitting/policy role, not a
   guarantee of useful precision or safety. Count connected groups once and report
   mixed-label groups explicitly. H=1 meets neither statistical needs nor this floor.
4. Reliability-policy fitting uses a separate calibration-policy partition with
   no shared patients/duplicate groups with calibration-fit. Prespecify risk endpoint
   (including positive-class false negatives), acceptable risk budget, one-sided
   confidence level, class/multiple-policy correction and finite candidate policy
   family **before fitting**. Evaluate per-class accepted counts and exact binomial
   bounds at the independent group level, with a documented rule for mixed groups.
   No accepted risk budget has been supplied, so no optimizer or clinical acceptance
   claim is enabled. Unsupported classes force REVIEW REQUIRED for full eight-label
   triage; do not borrow majority-class certainty for H. Two calibration roles cannot
   be justified by splitting the sole H-positive calibration patient. Cross-fitting
   would require a new preregistered dependence-aware protocol, not silent reuse.
5. Final evaluation remains sealed for model-result selection. Release it only
   after checksums freeze checkpoint, preprocessing, disease thresholds, calibrator,
   reliability policy and split/protocol. Assess once under that frozen pipeline;
   report all eight classes, primary patient metrics, secondary eye metrics,
   support, undefined statistics and component-bootstrap uncertainty. No subsequent
   seed/backbone/threshold selection from its results. The current evaluator refuses
   final release even with a freeze manifest: the complete calibrated final pipeline
   executor belongs to the later approved inference/evaluation phase.
6. For sparse classes report exact numerator/denominator and exact binomial bounds
   for sensitivity/specificity plus component dependence, not just macro averages.
   With one H-positive final patient, even one success has two-sided exact 95%
   interval [.025,1]. No reliable H calibration/sensitivity claim follows. Broader
   data, materially better duplicate adjudication or an explicitly narrower research
   estimand may be needed. Do not hide this by selecting partitions on performance.

The fit guard requires an explicitly enabled call and a trusted, checksum-bound
approval record whose duplicate review, split approval, final holdout lock,
calibration plan and limitations review are all true, plus role-specific immutable
manifest hashes. No such record exists. Reliability fitting and full final-pipeline
release remain explicit blockers even after that general guard.

**Training is not ready.** Material duplicate uncertainty, split approval,
rare-class support/power, calibration-role assignment, risk-budget definition,
verified pretrained sources and final-pipeline release remain unresolved. This
implementation performed no weight updates, calibration fitting, training runs,
GPU smoke tests or paid-resource execution.
