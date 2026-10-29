# Final held-out research evaluation

Generated from checksum-verified prediction and evaluation artifacts under the protocol committed before final exposure. No training, refitting, threshold tuning, strategy selection or final reporting-case selection occurred.

**Primary endpoint — raw patient-equal macro AUROC: 0.8580 [0.8366, 0.8798].**
All **350 patients / 700 eyes / 346 frozen containment components** were evaluated. Brackets denote fixed-seed 95% component-cluster percentile intervals, conditional on the frozen candidate. Secondary intervals are not multiplicity-adjusted clinical guarantees.

## Patient-level class results

Raw probabilities; fixed p >= 0.5. Positive shared temperature preserves every binary decision, so calibrated precision/recall/F1/accuracy are identical. Both raw/calibrated ranking metrics and both weighting schemes are in the CSV/JSON, including all undefined flags and replicate counts.

| Class | Positive / negative | AUROC [95%] | AUPRC/AP [95%] | Precision | Recall | F1 [95%] |
|---|---:|---|---|---:|---:|---|
| N | 111 / 239 | 0.7619 [0.7198, 0.8161] | 0.5547 [0.4680, 0.6642] | 0.5244 | 0.7748 | 0.6255 [0.5588, 0.7004] |
| D | 113 / 237 | 0.7729 [0.7157, 0.8190] | 0.5998 [0.5199, 0.6842] | 0.4528 | 0.8496 | 0.5908 [0.5413, 0.6508] |
| G | 22 / 328 | 0.9052 [0.8450, 0.9542] | 0.5184 [0.2869, 0.7093] | 0.5455 | 0.5455 | 0.5455 [0.3421, 0.7043] |
| C | 21 / 329 | 0.9810 [0.9599, 0.9949] | 0.8384 [0.6903, 0.9578] | 0.3443 | 1.0000 | 0.5122 [0.3703, 0.6238] |
| A | 16 / 334 | 0.8640 [0.7727, 0.9469] | 0.3901 [0.1661, 0.6306] | 0.2105 | 0.5000 | 0.2963 [0.1198, 0.4212] |
| H | 10 / 340 | 0.8806 [0.8159, 0.9333] | 0.2032 [0.0816, 0.5294] | 0.0763 | 0.9000 | 0.1406 [0.0702, 0.2222] |
| M | 18 / 332 | 0.9926 [0.9853, 0.9994] | 0.9231 [0.8453, 0.9912] | 0.4500 | 1.0000 | 0.6207 [0.4513, 0.7419] |
| O | 98 / 252 | 0.7058 [0.6411, 0.7653] | 0.4509 [0.3595, 0.5609] | 0.4673 | 0.5102 | 0.4878 [0.3897, 0.5727] |

## Overall classification

| Predictor / weighting | Macro AUROC [95%] | Macro F1 [95%] | Micro F1 [95%] | Exact-match accuracy | Labelwise accuracy | Hamming loss |
|---|---|---|---|---:|---:|---:|
| raw / patient_equal | 0.8580 [0.8366, 0.8798] | 0.4774 [0.4335, 0.5150] | 0.5124 [0.4847, 0.5420] | 0.1600 | 0.7961 | 0.2039 |
| raw / component_equal | 0.8586 [0.8387, 0.8800] | 0.4761 [0.4317, 0.5162] | 0.5124 [0.4859, 0.5412] | 0.1570 | 0.7958 | 0.2042 |
| calibrated / patient_equal | 0.8580 [0.8366, 0.8798] | 0.4774 [0.4335, 0.5150] | 0.5124 [0.4847, 0.5420] | 0.1600 | 0.7961 | 0.2039 |
| calibrated / component_equal | 0.8586 [0.8387, 0.8800] | 0.4761 [0.4317, 0.5162] | 0.5124 [0.4859, 0.5412] | 0.1570 | 0.7958 | 0.2042 |

Macro F1 averages eight class F1s. Micro F1 uses weighted TP/FP/FN on the (n,8) arrays. Exact match requires all eight labels correct; labelwise accuracy averages correct binary labels; Hamming loss is its complement. Any-label error is one minus exact-match accuracy. Precision/recall/F1 zero denominators use zero with flags; ranking requires both outcomes and macro AUROC requires all eight classes.

## Frozen calibration on held-out patients

The reused shared temperature is **T=0.722602142**. It acts on logit(clipped mean eye probabilities), not mean eye logits. No parameter or normalizer was refitted.

| Predictor / weighting | NLL [95%] | Brier [95%] | Macro ECE [95%] |
|---|---|---|---|
| raw / patient_equal | 0.4280 [0.4120, 0.4420] | 0.1397 [0.1336, 0.1452] | 0.1873 [0.1819, 0.1951] |
| raw / component_equal | 0.4291 [0.4131, 0.4425] | 0.1402 [0.1342, 0.1451] | 0.1877 [0.1823, 0.1958] |
| calibrated / patient_equal | 0.4196 [0.3970, 0.4380] | 0.1379 [0.1303, 0.1440] | 0.1668 [0.1615, 0.1772] |
| calibrated / component_equal | 0.4208 [0.3997, 0.4392] | 0.1384 [0.1308, 0.1444] | 0.1672 [0.1621, 0.1777] |

Paired differences are **calibrated minus raw**, on identical component resamples. Negative values favor calibration for these losses.

| Weighting | Δ NLL [95%] | Δ Brier [95%] | Δ macro ECE [95%] |
|---|---|---|---|
| patient_equal | -0.008465 [-0.013116, -0.004062] | -0.001802 [-0.003052, -0.000733] | -0.020534 [-0.022739, -0.015590] |
| component_equal | -0.008355 [-0.012841, -0.004233] | -0.001782 [-0.002975, -0.000666] | -0.020426 [-0.022799, -0.015415] |

Classwise paired calibration differences, patient-equal (calibrated minus raw):

| Class | Δ NLL [95%] | Δ Brier [95%] | Δ ECE [95%] |
|---|---|---|---|
| N | 0.003446 [-0.011109, 0.016551] | 0.002544 [-0.002403, 0.007176] | 0.002584 [-0.008946, 0.033338] |
| D | 0.019248 [0.003689, 0.034501] | 0.005419 [0.000756, 0.010543] | 0.006738 [-0.004361, 0.016858] |
| G | -0.031227 [-0.039534, -0.021689] | -0.005973 [-0.008082, -0.003055] | -0.035544 [-0.044231, -0.019143] |
| C | -0.033078 [-0.045010, -0.018950] | -0.006555 [-0.009766, -0.003665] | -0.040444 [-0.044764, -0.030362] |
| A | -0.021938 [-0.037330, -0.004135] | -0.005058 [-0.008234, -0.001579] | -0.040888 [-0.046784, -0.036801] |
| H | 0.024940 [0.006866, 0.048681] | 0.003492 [-0.000503, 0.008397] | -0.017925 [-0.022388, -0.013434] |
| M | -0.012259 [-0.022832, -0.000460] | -0.000762 [-0.002848, 0.001498] | -0.024168 [-0.028720, -0.020518] |
| O | -0.016849 [-0.022313, -0.010977] | -0.007519 [-0.010096, -0.004831] | -0.014622 [-0.023054, -0.001316] |

ECE uses ten fixed equal-width probability bins per class, weighted bin gaps and an equal mean across eight classes. It is sensitive to binning and sparse classes. [Patient reliability diagrams](patient_equal-reliability-diagrams.png) · [Component-equal diagrams](component_equal-reliability-diagrams.png).

## All fixed reliability ablations

Lower AURC is better. Full curves use every complete tied-score prefix and right-step integration. These are retrospective research ranking analyses, **not a validated automatic acceptance policy**. No useful-looking strategy or cutoff is selected.

| Fixed strategy | Weighting | Hamming AURC [95%] | Any-label-error AURC [95%] | Δ Hamming AURC vs confidence [95%] |
|---|---|---|---|---|
| raw__confidence | patient_equal | 0.164433 [0.143138, 0.184483] | 0.723772 [0.646756, 0.793426] | reference |
| raw__confidence | component_equal | 0.165970 [0.145016, 0.185844] | 0.733194 [0.656816, 0.804417] | reference |
| raw__confidence_uncertainty | patient_equal | 0.164565 [0.143181, 0.184643] | 0.723352 [0.646213, 0.792695] | 0.000132 [-0.000254, 0.000492] |
| raw__confidence_uncertainty | component_equal | 0.166099 [0.145214, 0.186006] | 0.732767 [0.656661, 0.804520] | 0.000129 [-0.000279, 0.000514] |
| raw__confidence_uncertainty_quality | patient_equal | 0.191433 [0.172209, 0.210116] | 0.818617 [0.761967, 0.876298] | 0.026999 [0.007517, 0.042379] |
| raw__confidence_uncertainty_quality | component_equal | 0.192287 [0.173561, 0.210347] | 0.822965 [0.767823, 0.881353] | 0.026317 [0.006790, 0.041865] |
| calibrated__confidence | patient_equal | 0.164433 [0.143138, 0.184483] | 0.723772 [0.646756, 0.793426] | reference |
| calibrated__confidence | component_equal | 0.165970 [0.145016, 0.185844] | 0.733194 [0.656816, 0.804417] | reference |
| calibrated__confidence_uncertainty | patient_equal | 0.164581 [0.143177, 0.184796] | 0.723614 [0.646664, 0.792848] | 0.000148 [-0.000350, 0.000541] |
| calibrated__confidence_uncertainty | component_equal | 0.166105 [0.145286, 0.185912] | 0.732977 [0.656738, 0.804859] | 0.000135 [-0.000364, 0.000554] |
| calibrated__confidence_uncertainty_quality | patient_equal | 0.189610 [0.170396, 0.208314] | 0.813015 [0.754185, 0.872651] | 0.025176 [0.006161, 0.040493] |
| calibrated__confidence_uncertainty_quality | component_equal | 0.190619 [0.172061, 0.208534] | 0.817902 [0.760529, 0.877233] | 0.024649 [0.005910, 0.039504] |

### Actual patient coverage at the saved development cutoffs

Nominal targets are validation coverage. The same saved score thresholds are applied unchanged; actual final coverage can differ. No final quantile is selected as an operating point. At nominal zero, no patients are retained and risk is undefined. The all-patient endpoint is separately reported above.

| Validation target | Raw C | Raw C+U | Raw C+U+Q | Cal C | Cal C+U | Cal C+U+Q |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.10 | 0.111 | 0.111 | 0.134 | 0.111 | 0.117 | 0.129 |
| 0.20 | 0.223 | 0.211 | 0.223 | 0.223 | 0.211 | 0.209 |
| 0.30 | 0.314 | 0.317 | 0.326 | 0.314 | 0.317 | 0.349 |
| 0.50 | 0.486 | 0.486 | 0.526 | 0.486 | 0.486 | 0.523 |
| 0.75 | 0.726 | 0.726 | 0.783 | 0.726 | 0.729 | 0.771 |
| 1.00 | 1.000 | 0.997 | 1.000 | 1.000 | 0.997 | 1.000 |

[All-eight retained support, false-negative counts/rates, precision/recall/F1 and actual weighted coverage](saved-grid-support-and-errors.csv). FNR/sensitivity are undefined with no positive support; empty-subset risk is undefined. Hamming risk can conceal negative-label imbalance; any-label-error curves are reported alongside it.

[Patient Hamming curves](patient_equal-hamming_risk.png) · [Patient any-label-error curves](patient_equal-any_error_risk.png) · [Component Hamming curves](component_equal-hamming_risk.png) · [Component any-label-error curves](component_equal-any_error_risk.png).

## Contribution and limitations

The study demonstrates a reproducible, frozen multilabel research pipeline with auditable patient/group separation, calibration, trained-head variability, technical quality and explanation interfaces. It evaluates one selected checkpoint; it cannot establish superiority over untested final-cohort alternatives.

Calibrated-minus-raw patient NLL is lower with a paired interval below zero on this cohort.

Pooled calibration gains were heterogeneous. Classwise NLL paired intervals were below zero for G, C, A, M, O; above zero (worse) for D, H; and included zero or were undefined for N. These are prespecified secondary, non-simultaneous intervals; no class-specific recalibration was performed.
Adding head uncertainty (raw) versus confidence-only: Hamming AURC has a paired interval including zero; a clear improvement is not established.
Adding technical quality to head uncertainty (raw) versus confidence-only: Hamming AURC is higher with a paired interval above zero on this cohort.
Adding head uncertainty (calibrated) versus confidence-only: Hamming AURC has a paired interval including zero; a clear improvement is not established.
Adding technical quality to head uncertainty (calibrated) versus confidence-only: Hamming AURC is higher with a paired interval above zero on this cohort.

These secondary comparisons do not authorize changing the frozen pipeline. Clinical automatic triage, complete epistemic uncertainty, clinical quality grading and lesion localization are not established. Deployment remains **REVIEW REQUIRED for all 350 patients**; no ACCEPT threshold is fitted or introduced.

- One internal ODIR holdout, no external clinical validation
- Patient-derived source labels do not establish eye-specific diagnosis
- Rare classes and sparse retained subsets have limited precision
- Head-only Dropout omits backbone/model uncertainty
- Heuristic quality is not clinical gradability
- Bounded duplicate assay cannot exclude every shared identity
- Bootstrap does not propagate training/temperature-fit uncertainty
- No clinical acceptable-risk contract or automatic acceptance cutoff

Bootstrap intervals use 200 resamples, seed 62026, of whole frozen components. Original inverse-component-size weights are preserved on each repeated draw. Intervals condition on the fixed model/temperature and have limited tail precision. Valid/total replicate counts are explicit; fewer than 160 defined values yields an undefined interval.

## Provenance and next phase

The protocol/configuration was committed before final exposure. Frozen contract, selected checkpoint, temperature, normalizer, development cutoffs, complete final coverage, model-state invariance, sources/nonces and every retrieved artifact checksum passed verification. Private predictions, images/features, score thresholds and bootstrap arrays stay outside Git in pinned private Kaggle outputs. No failed/replaced attempt or refit is hidden.

[Full permitted aggregate results](aggregate-summary.json) · [All endpoints and intervals](endpoints-and-intervals.csv) · [Per-class metrics](per-class-metrics.csv) · [Run registry](run-registry.json) · [Protocol](../../docs/FINAL_EVALUATION_V7.md).

Next phase: package this same frozen inference pipeline into the demo app, retaining its research limitations and REVIEW REQUIRED default. Do not retrain, recalibrate or tune against these final results.
