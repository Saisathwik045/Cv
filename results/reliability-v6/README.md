# Phase 6 conditional development results

Selected ConvNeXt-Tiny epoch two. Final evaluation remained sealed. No backbone retraining or acceptance-policy fit.

Shared temperature **T=0.722602**, fitted only on calibration patients after mean eye probabilities. Calibration fit statistics are **in sample**, not generalization evidence.

| Role | Metric | Raw | Calibrated |
|---|---|---:|---:|
| calibration | component_equal_nll | 0.429812 | 0.419442 |
| calibration | patient_nll | 0.432373 | 0.423283 |
| calibration | brier | 0.141013 | 0.139007 |
| calibration | macro_ece_10_equal_width_bins | 0.187692 | 0.166149 |
| validation | component_equal_nll | 0.435095 | 0.426425 |
| validation | patient_nll | 0.434740 | 0.425896 |
| validation | brier | 0.142055 | 0.140104 |
| validation | macro_ece_10_equal_width_bins | 0.187753 | 0.164830 |

Validation already selected the backbone. These comparisons are conditional development descriptions. Lower AURC is better; no reliability strategy or cutoff is selected.

| Predictor / fixed strategy | Hamming AURC | Any-label-error AURC | Hamming AURC bootstrap 95% |
|---|---:|---:|---|
| calibrated__confidence | 0.168652 | 0.710292 | [0.153054, 0.183667] |
| calibrated__confidence_uncertainty | 0.168585 | 0.709882 | [0.152893, 0.183562] |
| calibrated__confidence_uncertainty_quality | 0.200936 | 0.795471 | [0.184411, 0.218077] |
| raw__confidence | 0.168652 | 0.710292 | [0.153054, 0.183667] |
| raw__confidence_uncertainty | 0.168612 | 0.709786 | [0.152862, 0.183644] |
| raw__confidence_uncertainty_quality | 0.202406 | 0.798910 | [0.186109, 0.220280] |

Intervals use 200 resamples of whole containment components, conditional on the chosen backbone and fixed fitted temperature. They are not selection-adjusted, sparse-class or clinical confidence guarantees.

| Label | Validation positive / negative patients | Calibration positive / negative patients |
|---|---:|---:|
| N | 175 / 350 | 116 / 234 |
| D | 169 / 356 | 113 / 237 |
| G | 32 / 493 | 21 / 329 |
| C | 32 / 493 | 21 / 329 |
| A | 25 / 500 | 16 / 334 |
| H | 16 / 509 | 10 / 340 |
| M | 26 / 499 | 17 / 333 |
| O | 147 / 378 | 98 / 252 |

MC Dropout works at the already-trained p=0.2 head only: twenty seeded passes, both eyes aggregated within each pass. Stochastic depth and normalization remain in eval. It does not measure all backbone uncertainty. Quality uses training-only feature ECDFs and is an experimental technical signal, not clinical gradability.

Grad-CAM: 4 validation patients, 8 maps, 0 zero maps. All predicted-positive labels are saved separately from the explanation target. Private overlays are canonical model-input explanations, not verified lesion locations.

**Deployment default: REVIEW REQUIRED for every patient.** No independent supported policy partition/risk contract exists, so no automatic acceptance cutoff is fitted. All three raw/calibrated ablations are retained, including negative findings.

[Full permitted aggregates](aggregate-summary.json) · [Hamming curves](hamming_risk.png) · [Any-label-error curves](any_error_risk.png) · [Pinned run registry](run-registry.json)

Private artifacts contain the fitted parameter/provenance, candidate manifest, predictions, pass arrays, quality normalizer/signals, complete tied-threshold curves, fit diagnostics and Grad-CAM images. Use STATUS.md recovery commands. Final evaluation requires a separate authorized task with this frozen research candidate.
