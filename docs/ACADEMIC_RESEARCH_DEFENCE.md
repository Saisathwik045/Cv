# Research evidence and critical analysis

TrustRetina is a completed development research implementation through V15, not
an independently validated clinical system. This chapter connects research
questions to evidence and explains what an examiner can reasonably assess.
It uses committed public aggregates only; no new training, calibration, inference,
patient review or historical final-cohort access is needed to read it.
All decisions remain **REVIEW REQUIRED**.

Read with the [final report](FINAL_RESEARCH_REPORT.md),
[all-eight-class comparison](ALL_EIGHT_CLASS_COMPARISON.md),
[phase audit](PHASE_COMPLETION_AUDIT.md) and
[presentation outline](FINAL_PRESENTATION_OUTLINE.md).
The dashboard's **Research study & viva** page presents the same evidence boundaries.

## 1. Research questions and defensible conclusions

| Question | Experiment/evidence | Recorded finding | Limitation | Defensible conclusion |
|---|---|---|---|---|
| RQ1: Which initial backbone performs best under the shared development design? | [V5 five-model comparison](../results/full-v5/comparison.json) | ConvNeXt-Tiny selected by validation macro AUROC | Validation selected the model; macro scores conceal class errors | A development candidate was selected, not clinically certified |
| RQ2: Do schedule, loss or threshold changes improve the reference? | [V9](../results/round2-v9/RESULTS.md), [V10](../results/threshold-v10/RESULTS.md), [V11](../results/nested-v11/DISCUSSION.md) | Schedule/class tradeoffs; threshold CV reduces misses with more FP; square-root weighting did not establish superiority | Different estimands and development exposure; V10 thresholds never deployed | Retain unfavorable evidence and separate exploratory policy work from frozen inference |
| RQ3: Does320 improve over224? | [V13 matched folds](../results/resolution-v13/DISCUSSION.md), [V14 frozen fits](../results/candidate-v14/RESULTS.md) | Aggregate ranking gains coexist with C/A/G/H harms | Conditional single-seed development evidence, not independent confirmation |320 is a research nomination, not a universal winner or demo replacement |
| RQ4: What changes after shared-temperature calibration? | [V15 aggregates](../results/calibration-v15/aggregate-results.json) | Slight NLL/Brier decreases; ECE worsens in all four contexts; fixed-threshold classification unchanged | Calibration is in sample; validation was reused; rare-class support is sparse | Scalar loss improvement does not establish better discrimination or uniform reliability |
| RQ5: Do uncertainty and quality improve error ranking? | [V15 reliability report](../results/calibration-v15/RESULTS.md) | Quality fusion worsens Hamming AURC; all paired resolution-difference intervals include zero | Head-only MC, technical proxies and conditional bootstrap | No reliability strategy is promoted to an automatic acceptance policy |

These questions cannot be collapsed into a single claim that the project
"improved accuracy." Each experiment has its own cohort, selection procedure,
training size and endpoint. The original historical final evaluation cannot be
reused as independent evidence for later V14/V15 candidates.

## 2. Architecture and design choices to defend

The implemented path is paired input → EXIF/RGB letterbox and normalization →
eight logits → sigmoid per eye → mean-eye patient probability → frozen shared
temperature and reliability signals → REVIEW REQUIRED → optional target Grad-CAM.

| Choice | Reason / evidence | Boundary to explain |
|---|---|---|
| Eight sigmoid outputs and BCE | Multiple source targets may be positive together | No softmax or single-class argmax substitute |
| Original ODIR patient-derived labels | Authoritative source target semantics | Inheritance by eyes is not a verified diagnosis of either eye; H/O/N need careful external mapping |
| Patient and component grouping | Both eyes and known containment groups stay together | Residual duplicate uncertainty and repeated development exposure still matter |
| Shared model/preprocessing comparison | Common factories, target contract and evaluation definitions | v5 loss-based checkpoint selection differs from later selection rules |
| Mean-eye probability before temperature | Frozen patient aggregation; avoids changing the estimand | Mean calibrated eyes or mean logits is a different predictor |
| One shared temperature | Low-dimensional component-equal NLL fit on reserved calibration role | Not a classwise reliability or clinical-risk guarantee; numerical support floors are not power calculations |
| TRAIN-only technical quality ECDFs | Avoid outcome-driven feature normalization | Technical rank is not clinical gradability; quality fusion's negative result is retained |
| Trained-head MC Dropout | Uses existing trained dropout with normalization/backbone in eval | Omits backbone uncertainty; lower remapped variance is not proof of less epistemic uncertainty |
| Grad-CAM | Target-specific sensitivity illustration for each eye | No lesion ground truth, causal explanation or correctness guarantee |

## 3. Decision trail and ablation map

| Stage | What changed / was assessed | Decision retained | What must not be inferred |
|---|---|---|---|
| V5 | Five backbones under the original contained design | Select ConvNeXt by declared development criterion | A selection estimate is not unbiased generalization |
| V6/demo | Reserved shared-temperature fit, MC, quality and explanations | Freeze original epoch2/T≈0.722602 demo; always review | Later temperatures do not silently replace this contract |
| V9 | Fixed backbone/schedule variants | Preserve all comparison rows | A run-specific gain is not a universal schedule effect |
| V10 | Component-aware threshold CV | Keep study exploratory; do not deploy thresholds | Cross-validation does not undo earlier checkpoint selection |
| V11 | Matched nested training-only loss study | Retain ratio BCE reference; no established square-root superiority | A weak result is not evidence of equivalence |
| V12 | Saved-evidence failure audit | Record hypotheses and unresolved H/O issues | No causal attribution or clinical relabeling |
| V13 | Matched224/320 resolution folds | Nominate320 for further research with harms retained | Conditional paired intervals are not independent confirmation |
| V14 | Freeze full-role224/320 candidates and saved outputs | Keep distinct identities and original demo | No winner promoted to deployment |
| V15 | One scalar per candidate and fixed reliability ablations | Retain loss gains, ECE/quality harms and unchanged decisions | Calibration alone does not improve disease discrimination |

The [phase audit](PHASE_COMPLETION_AUDIT.md) is the source-linked chronology.
This table is an interpretation map, not a new combined statistical analysis.

## 4. Findings, failures and falsifiable follow-up

| Observation | Supported explanation | Still a hypothesis | Targeted future test, not executed here |
|---|---|---|---|
| NLL improves while ECE worsens | The prespecified scalar objective is component-equal NLL, not ten-bin ECE | Which subgroup/bin mechanisms dominate outside this cohort | Preregister calibration endpoints and bin/support sensitivity on compatible untouched data |
|320 has C FP harm and A/G recall regressions | Recorded class confusion counts differ between candidates | Detail loss, image quality or annotation ambiguity as causes | One-factor matched ablation, same roles/selection and explicit class endpoints |
| H estimates are fragile | Only16 positive validation patients and patient-derived semantics | Whether rare support or target ambiguity contributes most | Establish authoritative label compatibility and prospective support/precision planning |
| Quality fusion worsens risk ranking | Fixed multiplication ranks this development cohort less favorably | Whether the proxy is mismatched to clinical quality | Independently annotated gradability study under a separately specified protocol |
| Head-MC variance changes after temperature | The transformation changes probability-scale dispersion | Whether epistemic uncertainty decreased | Prespecified uncertainty evaluation on suitable untouched data; never infer safety from variance alone |

Use the [all-eight-class table](ALL_EIGHT_CLASS_COMPARISON.md) to discuss FP, FN,
precision, recall, F1 and supports together. Neither a favorable macro score nor
one selected heatmap establishes broad superiority. Never inspect private cases
or introduce patient screenshots just to make the defence more persuasive.

## 5. Contribution, originality and authorship

ConvNeXt, sigmoid/BCE, shared-temperature scaling, MC Dropout and Grad-CAM are
existing methods. The defensible project contribution is their integrated
paired-eye implementation, guarded experimental workflow, explicit provenance,
controlled comparisons and critical analysis of positive and negative findings.
Do not claim a new architecture or clinical breakthrough without corresponding evidence.

The submitting student must complete this ownership record truthfully:

| Contribution | Student's actual work and decisions | Code/report/history evidence | External or AI assistance and verification |
|---|---|---|---|
| Data contracts and leakage controls | To be supplied by author | Data pipeline / contained-role protocols | To be supplied by author |
| Model pipeline and experiment execution | To be supplied by author | Versioned source and public execution records | To be supplied by author |
| Calibration/reliability interpretation | To be supplied by author | V6/V15 protocols and aggregate reports | To be supplied by author |
| Application, tests and reproducibility | To be supplied by author | App, tests, locks and CI | To be supplied by author |

Commit history alone does not establish individual authorship. Include correct
method/dataset citations, data access conditions and institutional AI-assistance
disclosures. Author, institution and supervisor identities are not invented here.

## 6. Reproducibility and engineering evidence

| Level | What another reviewer can check | What it does not establish |
|---|---|---|
| Public evidence | Committed aggregates, source/protocol digests, reviewed figures and this comparison | Access to or authentication of unavailable patient-level source evidence |
| Software | Pinned environment, configuration checks, synthetic CPU tests, fail-closed bundle checks | Clinical performance, native Mac behavior or training reproducibility |
| Restricted research repetition | Separately approved exact inputs, checkpoints, roles, provenance and execution | New independent confirmation on the same already-used cohorts |
| Independent confirmation | New lawful compatible cohort, untouched endpoints and prospective design | Guaranteed clinical safety or deployment eligibility |

Use [submission reproduction instructions](FINAL_SUBMISSION_CHECKLIST.md).
Focused new-page checks after installing the documented app/report dependencies:

```sh
CUDA_VISIBLE_DEVICES='' python -B -m unittest discover -s tests -p test_academic_hub.py -v
CUDA_VISIBLE_DEVICES='' python -B -m unittest discover -s tests -p test_submission_demo.py -v
git diff --check
```

The new reporting page verifies the committed aggregate SHA256 and reviewed
figure digests, presents recorded metrics and performs no fitting or inference.
Numerical values are not reconstructed from private rows or recomputed outcomes.
The earlier V14/V15 scientific source/protocol snapshots remain historical frozen
identities; a new reporting module is not a new scientific execution.

## 7. Deployment assessment and cloud lifetime

Previously committed [original demo verification](../results/demo-v8/verification.json)
and [demo protocol](DEMO_V8.md) record Linux CPU loading about1.020 seconds,
paired inference0.214–0.274 seconds, target-D Grad-CAM0.311 seconds and peak
process RSS844.6 MiB. These are historical single-process observations, including
verification arrays, not a Mac benchmark, sustained throughput or capacity guarantee.
The new academic page does not benchmark or rerun the private model.

| Item | Status / remaining evidence |
|---|---|
| Original model contract | Hash-bound epoch2,224 px, original temperature, fixed0.5 decisions |
| Input handling / privacy | Pair required, bounded decode, session cleanup, no random-weight fallback; reset is not secure erasure |
| Mac | Native ARM64 Python3.11/3.12 and macOS13+ packaging target; local rehearsal still required |
| Cloud preview | Loopback within the existing workspace only; dies when the runtime stops; no uptime or inbound URL guaranteed |
| Public deployment | No public hosting or clinical deployment performed; separate access/privacy/security approval would be required |
| Preservation | No confirmed durable encrypted backup/restore drill; cloud working files are not an archive |

The previous Mac handover archive is a frozen prepared package, not automatically
updated by this reporting change. Review/merge this source branch before expecting
the new page in a fresh main checkout. No private model transfer is authorized by
adding this page.

## 8. Viva and examiner walkthrough

1. State the problem, five distinct research questions and the development-only scope.
2. Draw the paired-eye architecture and explain multi-label targets, grouping and role separation.
3. Defend the selection rules and explain why v5, nested folds and later full-role fits cannot be pooled.
4. Show all eight classes, including C/A/G/H regressions and sparse supports.
5. Explain the negative V15 ECE and quality findings; distinguish objective loss from clinical benefit.
6. Explain head-only MC and Grad-CAM limitations without calling them safety guarantees.
7. Demonstrate safeguards with generated images only; identify fake-service tests versus permitted real-model smoke tests explicitly.
8. Show provenance/reproduction levels, truthful contribution disclosures and the precise independent-validation gap.

Expected questions: Why no softmax? Why mean-eye sigmoid first? Why component
weights? Why one temperature? Why unchanged F1? Why can calibration worsen ECE?
Why is higher resolution not uniformly better? What does a heatmap fail to prove?
What did you personally implement and verify? Which single next experiment would
most reduce uncertainty, and what would falsify your preferred explanation?

The [presentation outline](FINAL_PRESENTATION_OUTLINE.md) provides the main deck.
An excellent defence connects every answer to a specific source and admits what
has not been tested. It does not need another model, a favorable invented result,
or a patient-level example.
