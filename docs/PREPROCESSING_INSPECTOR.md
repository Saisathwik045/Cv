# Preprocessing inspector

Open **Preprocessing inspector** in the existing Streamlit dashboard. This
synthetic-only educational workspace explains the original frozen input pipeline
without a private bundle, checkpoint loading, model inference or patient uploads.
It is a functional visualization, not a new research result or clinical test.

## Walkthrough

1. Choose a generated landscape, portrait, square or wide frame and a left/right
   fixture colour. These shapes are visibly marked SYNTHETIC NON-PATIENT, not
   retinal photographs or disease examples. The colour selector is not anatomical
   laterality validation; the paired predictor still requires both real eye inputs.
2. Compare the decoded EXIF-corrected/RGB source with the actual **224 × 224 RGB
   canvas before normalization**. Both displays may be enlarged for visibility;
   the geometry table reports actual pixel dimensions.
3. Inspect the bilinear resized-content dimensions, left/top/right/bottom padding
   and fraction of the canvas occupied by padding. Landscape 320×240 becomes
   224×168 plus 28 pixels above/below; square 320×320 becomes 224×224 with no padding.
4. Optionally show the cyan content boundary. This is a separate annotated copy,
   never the canvas passed to preprocessing or the model.
5. Read the actual normalized float32 tensor's shape and per-channel minimum,
   maximum and mean. The input tensor is `[3,224,224]`; a real paired prediction
   stacks two images as `[2,3,224,224]`. No backbone is executed here.
6. **Reset inspector** returns to landscape/left/no overlay and clears only
   inspector controls. Widget generations change so the browser also drops stale
   selections. The sidebar reset still clears the entire app session.

## Exact implementation and identity

The inspector calls the existing trusted decoder, `letterbox_rgb` and `preprocess`
with the original `PreprocessConfig` and `role='inference'`. It checks the
configuration fingerprint against the committed original inference-v8 contract
before generating output. The fingerprint identifies configuration; source
integrity checks separately preserve the frozen implementation. No scientific
source under `src`, weights, temperature, threshold, normalizer, protocol or
recorded output is changed.

The frozen transformation is EXIF transpose/RGB → bilinear resize preserving aspect
ratio subject to integer rounding → mean-colour letterbox → RGB bytes/255 →
per-channel `(x-mean)/std`. There is no inference crop or random flip. Means are
`(0.485,0.456,0.406)` and standard deviations `(0.229,0.224,0.225)`.
Rounded padding bytes are `(124,116,104)`, so normalized padding is near zero,
not exactly zero. Negative normalized values are expected; the RGB preview does
not falsely render negative tensor values as colour pixels. A numerical check
requires tensor de-normalization to recover the displayed canvas within 1e-6
absolute float32 tolerance and geometry metadata to agree.

## Limits and privacy

This workspace has no file uploader, export, bundle path control or model-loading
action. Images/tensors are generated and processed in memory, with no application
disk writes or data caching. Browser/server media and session lifecycle are not
a secure-erasure guarantee. It does not read private research storage or alter
the separate manual prediction workspace. No new public deployment or model
transfer is introduced; cloud preview availability still depends on the runtime.

Padding percentage is a geometry observation, not retinal coverage, a quality
grade or an acceptance criterion. Shapes have no medical ground truth. Resizing
can lose detail, but this page does not quantify disease-detection impact or prove
retinal suitability, lesion localization, accuracy or clinical robustness.
Every prediction elsewhere remains REVIEW REQUIRED and independent clinical
validation remains incomplete.

## Verification

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  python -B -m unittest discover -s tests -p test_preprocessing_inspector.py -v
```

Synthetic tests compare all four presets with actual service preparation without
constructing a model, check padding, EXIF handling, invalid inputs, identity
refusal, overlay copy isolation, UI controls, scoped reset and sanitized failure.
Checkpoint loading/model construction are forbidden in the UI test. Missing
optional CPU dependencies are explicit skips in minimal environments; the full
CPU app environment is needed for complete coverage.

Native Chromium inspection additionally observed the rendered source/canvas,
geometry, overlay, normalization/statistics, fingerprint and square-frame change.
It exposed a stale-widget display issue; generation-based reset was then verified
in the browser to restore landscape/left/no overlay consistently. No real-model
inference ran. See [current verification status](STATUS.md) and
[cloud viewing boundaries](CLOUD_DEMONSTRATION.md).
