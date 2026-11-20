# Academic submission governance review

Review date: 20 November 2026. Scope: public documentation, synthetic software
checks and opaque-byte preservation verification. No new research analysis,
patient review, fitting, training, real-model inference or historical-final access.
Slides and poster preparation remain deferred at the student's request.

## Claim and citation review

The report describes completed development research, not an independently
validated clinical product. Its conclusions retain worse V15 ECE, negative quality
fusion findings, class-specific harms, reused validation and sparse support.
The original demo and later V14/V15 candidates remain distinct. Synthetic tests
and CPU timing establish software operation only. No new clinical superiority,
diagnostic accuracy, safety, gradability or lesion-localization claim is added.

Primary paper landing pages were opened and checked against the report's titles,
lead authors and identifiers on the review date:

| Method | Primary source | Review result |
|---|---|---|
| ResNet | He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | Title, attribution and identifier match |
| DenseNet | Huang et al., [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) | Title, attribution and identifier match |
| EfficientNet | Tan and Le, [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | Title, attribution and identifier match |
| ConvNeXt | Liu et al., [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) | Title, attribution and identifier match |
| Temperature scaling | Guo et al., [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | Title, attribution and identifier match; project aggregation/weighting is protocol-specific |
| MC Dropout | Gal and Ghahramani, [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) | Title, attribution and identifier match; head-only implementation has narrower scope |
| Grad-CAM | Selvaraju et al., [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) | Title, attribution and identifier match; no clinical localization validation inferred |

This checks bibliographic identity, not a systematic literature review, citation
style compliance or reproduction of each paper. These methods are prior work,
not inventions of TrustRetina. No external paper's benchmark numbers are imported
as project results.

## Rights, dataset conditions and ethics

| Item | Observed status | Required action |
|---|---|---|
| Repository software license | No tracked LICENSE/COPYING file was found in this review | Author/institution must select an appropriate license only after confirming ownership and third-party obligations; public visibility is not a license grant |
| ODIR access and redistribution conditions | The [official challenge homepage](https://odir2019.grand-challenge.org/) returned HTTP 403 to the lookup tool; rules could not be retrieved | Author supplies applicable access terms/license evidence from their authorized source; do not infer permission from Kaggle availability or invent a license |
| Dataset/model rights | Research artifact recovery is not proof of redistribution rights | Keep images, patient-level targets/predictions, manifests and checkpoints private; separately review model and pretrained-weight conditions before distribution |
| Institutional ethics determination | No approval/exemption decision established by this audit | Supervisor/institution determines applicable requirements and records its decision privately; do not claim approval, exemption or consent on their behalf |
| Privacy and data minimization | Public work uses aggregate evidence and synthetic inputs; private preservation checks hash bytes without parsing rows | Do not include patient screenshots, exports, approval records, credentials or private inventories in the submission repository |
| Assistance and ownership | Contribution template exists; student-specific completion is pending | Record actual student decisions/work and external/AI assistance truthfully; commit history or quiz agreement alone does not prove authorship |

The audit does not replace legal or institutional review. No license, consent,
DUA or institutional declaration has been accepted or signed by the assistant.
The use of existing methods should be acknowledged alongside the project's actual
integration, reproducibility controls and critical analysis contributions.

## Supervisor review and author sign-off form

This is a review form, **not an approval or completed declaration**. Do not fill
unknown fields with guessed identities. Keep signatures and private institutional
correspondence outside the public repository.

| Field | Current status |
|---|---|
| Author name / student identifier | PENDING — author supplied details required; publish identifiers only if institution requires and author permits |
| Institution / department / programme | PENDING — not supplied |
| Course/project title and submission deadline | PENDING — not supplied |
| Supervisor / assessor | PENDING — not supplied |
| Required report format, rubric and citation style | PENDING — not supplied |
| Actual contribution and assistance statement | PENDING — student must verify and complete the [contribution record](ACADEMIC_RESEARCH_DEFENCE.md) |
| Dataset access/rights and ethics determination | PENDING — supporting institutional/source records required |
| Report scientific limitations accepted | PENDING — supervisor review, not assistant certification |
| Required final slides/poster | DEFERRED — student requested later preparation |
| Submission commit and CI run | Record the reviewed final commit/run after PR review; do not treat an unmerged branch as main |
| Supervisor decision and date | PENDING — must be supplied by supervisor |

Suggested supervisor review questions:

1. Is development-only scope acceptable with independent clinical validation
   explicitly incomplete and historical final data unavailable for fresh confirmation?
2. Does the report accurately separate patient-derived targets, eye inheritance,
   selection, calibration diagnostics and reused-validation descriptions?
3. Are method/dataset citations, access conditions, ethics and assistance statements adequate?
4. Are the synthetic demo, failure handling, CPU observations and clean-environment
   software evidence sufficient for this course's technical requirements?
5. Which institutional formatting, author declarations and presentation deliverables
   remain before acceptance?

The technical audit can be completed autonomously. Institutional acceptance,
individual authorship declarations and a durable encrypted restore test cannot
be truthfully marked complete without their corresponding human evidence.
