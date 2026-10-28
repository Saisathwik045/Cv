# Persistent TrustRetina rules

- Follow the user's phase scope. Phase 0 must never launch training or paid resources.
- Continue each new phase from the latest remote `main`, with one `phase/NN-name`
  branch and one PR per phase. Do not merge without user authorization.
- Read docs/PROJECT_CONTEXT.md, docs/STATUS.md and docs/EXPERIMENT_PROTOCOL.md first.
- Use Python scripts and configuration; no manual notebook cell execution or
  hardcoded Kaggle paths. Notebooks are reference-only.
- Preserve: image → preprocessing → 8-label classifier → confidence + MC Dropout
  uncertainty + quality → calibration → ACCEPT / REVIEW REQUIRED → Grad-CAM.
- Use original ODIR-5K `data.xlsx` as authoritative labels and class order
  N,D,G,C,A,H,M,O. These are patient-derived labels, not verified eye labels.
- Keep both eyes together and reserve train/validation/calibration/final_evaluation
  patients before training. Check positive and negative support per class/partition.
- Never select models, disease thresholds or triage policies on final evaluation
  labels. Never compute official-test metrics without ground truth.
- Compute multilabel F1 on equal-shaped two-dimensional (n,8) binary arrays.
  No argmax, flattened arrays or softmax for this multi-label task.
- Never fabricate results, manually fill comparison tables, or force reliability,
  image quality or calibration to improve results. Negative results are valid.
- Raw inputs are immutable. Never commit datasets, patient manifests, predictions,
  images, checkpoints, credentials or tokens. Do not print secrets.
- Only expose working commands. Unimplemented operations must be absent or fail
  nonzero with an explicit explanation; planning must never claim execution.
- Reuse completed stages only after verifying artifact hashes and full provenance.
- Run lightweight checks documented in README and git diff --check before committing;
  add meaningful tests for research-critical functionality. Update docs/STATUS.md.
- GPU/account/network readiness requires observed evidence; never infer GPU access
  from being in a cloud runtime or from credentials merely being present.
