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
training settings and reliability ablations. The initial foundation/data contracts
were followed by the contained v4 release,
completed five-backbone comparison, original reliability/evaluation/demo and
later development experiments through real V15 CPU calibration. The current
[final report](FINAL_RESEARCH_REPORT.md) and [phase audit](PHASE_COMPLETION_AUDIT.md)
separate historical planning from observed completion. Independent clinical
validation and broad superiority/safety claims remain unsupported.

Validation performed phase-specific checkpoint/backbone selection. Exploratory
v10 threshold CV was not deployed. Calibration fits a shared scalar on its reserved
role; no validated acceptance policy is fitted and all decisions remain REVIEW
REQUIRED. Historical final evaluation was frozen and executed once, is now used,
and cannot provide fresh confirmation for later candidates. Official-test images
without ground truth cannot provide evaluation metrics. Current submission scope
uses public aggregates and synthetic software checks, not private cohort access.
