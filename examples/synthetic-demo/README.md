# Generated non-patient demonstration inputs

The setup command generates three small PNGs locally, not in the repository:
`~/Library/Application Support/TrustRetina/demo-images`. They are geometric software
fixtures, not retinal photographs or patient data. Their visible text says
SYNTHETIC NON-PATIENT. They have no disease ground truth; model outputs on them
have no medical meaning. Image files are deliberately not committed.

After the separately permitted original model is installed, upload `left.png`
and `right.png` to demonstrate operational wiring, eight outputs, REVIEW REQUIRED,
signals, export and Grad-CAM. Replace the left upload with `replacement.png` to
check that old results clear, then reset the session. Do not use these fixtures
to claim model accuracy, localization validity, calibration quality or clinical safety.

No expected disease labels or desired prediction values are supplied. To generate
them manually after the app environment is installed, use:

```sh
.venv-mac/bin/python scripts/generate_demo_images.py \
  --destination "$HOME/Library/Application Support/TrustRetina/demo-images"
```
