# Project context

TrustRetina builds an automated, reproducible retinal disease screening research
pipeline with minimal manual work. Source repository: Saisathwik045/Cv, initially
only README and .gitignore; the other attached repository is an unrelated compiler.
There are no legacy notebooks or datasets in this repository at Phase 0.

The architecture, five backbones and N,D,G,C,A,H,M,O ordering are fixed as described
in README. Labels originate at patient level in original ODIR-5K `data.xlsx`.
The initial research unit is an eye image carrying patient-derived labels. Report
that limitation and patient clustering honestly; future patient-level aggregation
must be separately specified before inspecting final results.

Configuration separates data location, artifacts, partition fractions, seeds,
training settings and reliability ablations. Phase 0 builds checks and designs the
execution contract; Phase 1 implements data validation, grouped split bundles and runner checks with
synthetic-only validation. Remaining phases implement model execution. No medical result
or superiority claim is supported yet.

Validation selects model weights and disease thresholds. Calibration fits
probability calibration and the acceptance policy under a prespecified budget.
Final evaluation is isolated until model, thresholds, calibration and policy freeze.
Official test images without ground truth are prediction-only inputs.
