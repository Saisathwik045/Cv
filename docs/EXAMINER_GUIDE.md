# Examiner guide and assessment rehearsal

TrustRetina is a development research project through V15. The assessment should
reward understanding, reproducible evidence and honest interpretation. A working
dashboard is one deliverable; it does not establish clinical validity.
All predictions remain **REVIEW REQUIRED**. This guide introduces no new study,
model, calibration, evaluation or institutional assessment policy.

## What a teacher should expect

The following is a **proposed discussion rubric**, not an official institutional
scheme or an awarded mark. Adapt the weights with the supervisor. No marks have
been assigned and no self-test result determines submission readiness.

| Area | Suggested maximum | Evidence to demonstrate | What a strong defence explains |
|---|---:|---|---|
| Problem and target semantics | 10 | [Report](FINAL_RESEARCH_REPORT.md) | Research question, multi-label targets, patient-derived eye-label limitation and intended research use |
| Architecture and implementation | 15 | [Design defence](ACADEMIC_RESEARCH_DEFENCE.md) and original demo contract | Trace two images through preprocessing, sigmoid, mean-eye aggregation, temperature, signals and explanations; justify alternatives |
| Data integrity and experimental design | 20 | [Protocol](EXPERIMENT_PROTOCOL.md) and [phase audit](PHASE_COMPLETION_AUDIT.md) | Patient/component role separation, model selection, calibration role, duplicate limitations and validation reuse |
| Evaluation and critical analysis | 20 | [V15 report](../results/calibration-v15/RESULTS.md) and [class comparison](ALL_EIGHT_CLASS_COMPARISON.md) | Distinguish calibration diagnostics from reused validation; explain class harms, worse ECE and quality fusion, support and uncertainty |
| Software and reproducibility | 15 | [Submission checks](FINAL_SUBMISSION_CHECKLIST.md), locks, tests and current PR CI | Reproduce public evidence/software checks; distinguish mocks, actual operational tests and scientific evaluation |
| Demonstration and communication | 10 | [Cloud walkthrough](CLOUD_DEMONSTRATION.md) and [slides outline](FINAL_PRESENTATION_OUTLINE.md) | Eight independent outputs, REVIEW REQUIRED, synthetic labels, heatmap limits, export/reset and honest fallback |
| Ownership, ethics and scholarship | 10 | [Contribution record](ACADEMIC_RESEARCH_DEFENCE.md) and final references | Actual student decisions, third-party/AI assistance, citations, privacy, data terms and limits of claims |
| Total proposed | 100 | Examiner/supervisor judgement only | No automated grade or claim of clinical competence |

## Ten-minute evidence-led demonstration

| Time | Action | Examiner challenge |
|---|---|---|
| 0–1 min | State the research question and intended use | Why are eight outputs multi-label rather than eight mutually exclusive diagnoses? |
| 1–3 min | Trace the architecture and explain split/selection roles | Why keep both eyes/components together? What does repeated validation compromise? |
| 3–5 min | Show recorded V15 tables and a class-specific limitation | Why can NLL decrease while ECE worsens? Why does fixed-threshold classification stay unchanged? |
| 5–7 min | Run local readiness, then the labelled synthetic walkthrough if available | What is actually verified? Why are generated-image outputs not evidence of disease performance? |
| 7–8 min | Show both-eye explanation, JSON context, replacement and reset | Why is Grad-CAM not verified lesion localization? What clears and what remains downloaded? |
| 8–9 min | Identify a negative result and its next falsifiable experiment | Separate an observation from a hypothesized cause; do not propose tuning on the used final cohort |
| 9–10 min | Show reproducibility boundary and ownership statement | What did you implement/verify yourself? What independent confirmation remains missing? |

Use **Examiner rehearsal** in the dashboard for six no-model reasoning scenarios.
Choices have teaching-key feedback and repository evidence references. Answers
are session-local; the optional JSON contains topic feedback only, with no name,
free-text response, patient data, research predictions or grade. The page does
not assess oral reasoning, individual authorship or clinical competence. Reset
clears only rehearsal state; sidebar reset clears the app session. Downloads
remain on the downloading device until the user manages them.

## Failure rehearsal and honest fallback

| Situation | Required response |
|---|---|
| Bundle absent/corrupt | Show the fail-closed readiness message. Do not replace weights, invent outputs or download automatically. Use the public research page to discuss recorded findings. |
| Cloud route/session unavailable | State that the live demo is unavailable. Present committed report/figures; do not label recorded evidence as a current live prediction. Rehearse the actual presenting route beforehand. |
| Only one eye supplied | Explain that the manual predictor requires a real pair; never duplicate an eye to create a patient pair. Use generated fixtures only in their explicitly labelled workspace. |
| Attractive heatmap or high probability | State that this is a sensitivity illustration/probability, not a verified diagnosis, safety guarantee or lesion boundary. |
| Question about universal improvement | Show negative ECE/quality findings and class-specific tradeoffs; do not infer independent superiority from development results. |
| Question about permanent hosting/backup | Separate temporary cloud operation from durable storage; no uptime or encrypted-backup claim without observed verification. |

## Submission evidence and human sign-off

| Deliverable | Existing evidence | Remaining human action |
|---|---|---|
| Research narrative | Final report, phase audit and public result reports | Check references, final prose and institutional PDF requirements |
| Presentation | Slide outline, evidence hub, cloud demo and rehearsal | Build institution-formatted slides; rehearse timing and questions |
| Reproducibility | Pinned environments, synthetic tests, public hash checks and CI | Record exact submission commit and test-run URL; restricted research reproduction needs separate approvals |
| Ownership | Contribution/assistance template in research defence | Supply truthful student contributions, author, institution, supervisor and assistance disclosures |
| Academic acceptance | Development-only limitations documented | Obtain supervisor review and institutional sign-off; no automatic pass claimed |
| Scientific confirmation | Independent validation explicitly incomplete | Preregister untouched compatible data, label mapping, overlap controls, endpoints and support/precision plan before separately authorized execution |
| Private preservation | Separate backup requirements | Approve destination, encryption recipient and transfer; verify a full restoration before claiming durable backup |

Native Mac testing is optional for the chosen cloud-only demonstration, not a
reason to claim the cloud workflow is blocked. Neither cloud verification nor
this rehearsal establishes native Mac compatibility. Independent confirmation
is a research gap to disclose, not a prerequisite for honestly submitting a
development-only academic project if the institution accepts that scope.

## Proposed follow-up, not executed

For measured implementation cost, use the [CPU performance report](CPU_PERFORMANCE.md).
It records a bounded original-model run on generated non-patient images, not
clinical accuracy, browser latency, concurrency capacity or native Mac performance.

The most valuable scientific extension is compatible independent confirmation,
not more interface decoration. It requires data access and a prospective design.
A clinical-quality annotation study could test whether technical proxies measure
gradability, but it needs appropriate authorized images/annotations. A matched
ablation could test a specific class-error hypothesis, with fixed selection rules
and endpoints. None of these studies, their results or approvals are implied by
adding this examiner pack. Do not rerun fitting, inference or historical final
evaluation as part of rehearsal.
