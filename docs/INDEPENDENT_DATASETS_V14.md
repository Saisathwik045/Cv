# Independent evaluation feasibility — v14

Reviewed 10 November 2026 from primary documentation. The bounded search covered
five external options plus the ODIR creator description. **No verified independent
paired-eye cohort with compatible ground truth for all eight ODIR labels was
identified.** This does not establish that no such resource exists anywhere.
[Machine-readable source catalog](../results/candidate-v14/dataset-options.json)
records documentation access, versions, unknowns and the fact that no external
patient files were downloaded. Missing labels must remain unknown.

## Documented options

| Option and authoritative sources | Pairing / identity evidence | Label and access feasibility | Decision |
|---|---|---|---|
| [BRSET 1.0.2, official release](https://physionet.org/content/brazilian-ophthalmological/1.0.2/) and [license](https://physionet.org/content/brazilian-ophthalmological/view-license/1.0.2/) |16,266 images/8,524 patients; `patient_id` and `exam_eye`. Complete-pair count is unverified. Version 1.0.2 explicitly reviews eye laterality. | Native retinal findings include DR, AMD, myopic fundus and hypertensive retinopathy; no cataract target documented. Credentialed Health Data License 1.5.0, DUA and CITI training. | Leading **partial-label feasibility** option; neither eight-label compatibility nor restricted-data entitlement established. |
| [PAPILA author-institution record](https://portalinvestigacion.upct.es/documentos/668fc47cb9e7c03b01bdee3a?lang=en), [release v2](https://doi.org/10.6084/m9.figshare.14798004.v2), [primary descriptor](https://www.nature.com/articles/s41597-022-01388-1) | Primary article indexed text documents488 eyes/244 patients, patient IDs and OD/OS filenames. | Glaucoma/healthy/suspect; no complete eight-label panel. Institution records GPL 3.0+ for the dataset; confirm actual release terms before use. Public release documented, actual files not verified. | Possible paired-eye G-only transport study; suspects stay indeterminate. |
| [RFMiD primary descriptor](https://www.mdpi.com/2306-5729/6/2/14), [official download](https://riadd.grand-challenge.org/download-all-classes/) |3,200 images/46 conditions reported; image ID documented, patient/laterality linkage not verified. | Descriptor states dataset CC-BY 4.0. Download page403; exact archive/version/access terms pending. Media haze and optic-disc cupping cannot substitute for cataract and glaucoma. | Not eligible for the fixed paired-patient endpoint without verified linkage; possible separately described image-level partial study. Never manufacture a pair. |
| [mBRSET 1.0 official record](https://physionet.org/content/mbrset/1.0/) |5,164 images/1,291 patients; `patient`/`laterality`, multiple views. | DR grades/edema and self-reported systemic hypertension; not an all-eight panel. PhysioNet credentialing/training/DUA required; no entitlement verified. | Later DR/mobile-domain option. Systemic hypertension is not automatically an ODIR H match. |
| [EyePACS2015 official data](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data) and [rules](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/rules) | Official indexed text describes subject IDs and left/right suffixes. Actual complete labeled pairs unverified. | Clinician DR grades 0–4 only; subject to competition rules. Current full rules and account acceptance not verified. Existing Kaggle credentials do not establish entitlement. | DR-only fallback. Grade 0 is not an all-eight normal label. No competition enrollment or terms acceptance performed. |

Direct PhysioNet descriptions/licenses, the author-institution PAPILA record,
RFMiD descriptor and creator repositories were readable. Nature/PMC/Figshare
opens were blocked/challenged; PAPILA pairing/diagnostic details above use indexed
**primary article** text, with that limitation recorded. Kaggle pages rendered no
readable body; indexed official data/rules text was available. No unofficial
mirror was substituted. Access permissions, exact downloaded bytes and metadata
integrity still need checking before patient-level use; documentation review does
not establish actual usable sample counts.

## Compatibility is an endpoint question

ODIR's [creator description](https://github.com/nkicsl/OIA-ODIR) defines patient
labels using both eyes and age: N, diabetes D, glaucoma G, cataract C, AMD A,
hypertension H, myopia M and other diseases/abnormalities O. It does not provide
an exhaustive H/O case-definition or keyword-to-label algorithm. [Prior v13
semantics](RESOLUTION_V13.md) remain unresolved. The official challenge page
returned403; the creator repository was readable. No relabeling is proposed. Without authoritative annotation guidance sufficient
to resolve H/O and D, even a new labeled cohort cannot be claimed to reproduce
ODIR semantics; retain explicitly named native-label transport endpoints instead.

| ODIR target | What must be resolved before external confirmation |
|---|---|
| N | A healthy glaucoma eye, DR grade 0 or normal anatomy does not establish absence of all eight conditions. Require explicit complete-panel normal criteria. |
| D | ODIR says diabetes; external DR grading is a retinal endpoint. Do not silently equate them. A separately named head-D-versus-native-DR transport endpoint may be registered, with its mismatch disclosed. |
| G | PAPILA's clinical glaucoma is a possible endpoint after definition review. Suspect is not negative. Increased cup-disc ratio/optic-disc cupping alone is not a glaucoma diagnosis. |
| C | RFMiD media haze is not a cataract diagnosis; BRSET lacks a documented cataract field. Missing C is unknown. |
| A | External AMD is a promising conceptual match, subject to grading criteria, laterality and patient-level aggregation review. |
| H | ODIR hypertension, BRSET hypertensive retinopathy and mBRSET self-reported systemic hypertension are different documented targets. Their equivalence is not established; never infer a diagnosis from the images. |
| M | Myopic fundus/pathologic myopia and general myopia can differ in criteria. Refractive measurements alone do not authorize a new ODIR label. |
| O | An unrelated “other” field, union of convenient findings or complement of missing classes cannot reproduce an unspecified residual category. Require explicit inclusion/exclusion rules. |

Without verified patient linkage, RFMiD image-only results would be descriptive;
copy-component clustering cannot establish patient independence or valid patient-
cluster intervals. It is excluded from the paired-patient primary endpoint.

No mapping is currently approved. A qualified source/schema review, performed
without model predictions, must register each native endpoint and its known-label
mask. That review is not clinical adjudication by this agent. For partial data,
report the native endpoint (for example head-A score versus BRSET AMD), and a
clearly named partial-label summary over its registered classes. Do not call it
all-eight ODIR validation. All unassessed targets stay unknown, including targets
not recorded for “negative” patients. If confirmed G in one eye establishes a
registered patient G endpoint, OR may be used; a patient negative requires known
negative assessment in both eyes. Conflicting/missing/suspect findings remain
explicit rather than being silently resolved.

## Independence and overlap gates

The documented Brazilian, Spanish and Indian collections are plausible external
sources relative to Chinese ODIR; geography alone does not prove non-overlap.
No external patient files were acquired here, so **no content-level non-overlap
claim has been established**. BRSET/mBRSET are related projects; do not count them
as independent replications without checking cross-cohort overlap.

For a future permitted acquisition, pin the release, license/DUA, files and SHA256;
verify source IDs, eye laterality, visits, pair completeness, repeated views,
missing-label coding and conflicts before inference. Use one complete exam per
patient. Prefer source-designated macula-centered views and earliest documented
exam; break residual ties by lexicographic source image ID. If necessary metadata
is absent, register the deterministic fallback and field-of-view limitation before
seeing predictions; never choose the pair yielding a better model result.

Screen new external images against existing frozen ODIR byte/pixel hashes and
available content fingerprints, plus within/across external sources. Recover
existing all-ODIR hash records if available; do not rerun completed ODIR duplicate
investigation or open historical final images/labels. The train-only v13 hashes
alone cannot establish exclusion of all prior ODIR exposure. If complete prior
hash/fingerprint coverage is unavailable, report that blocker.

Weak dHash alone is not a copy decision. Any new material candidates require
independent retinal-content/alignment evidence under the existing fixed technical
method, with a bounded review list and manifest. Confirmed/probable copies and
within-external containment links form indivisible components. An external group
linked to ODIR cannot be an independent evaluation group: exclude it from that
primary cohort with a recorded reason. Material unresolved cross-source cases
receive targeted primary exclusion/sensitivity treatment, not blanket removal of
thousands of weak matches. Preserve raw data/labels and exclusion counts. If
verification would require prohibited historical-final pixels, stop/contain that
case rather than open them. This procedure is a proposed new-cohort overlap gate,
not a repeat of completed ODIR adjudication and not a patient re-identification
attempt.

## What fresh all-eight evidence would require

A collaborator-held or newly consented cohort from suitable independent sites/time,
with native paired color fundus images, stable pseudonymous patient/exam/laterality
keys, clear repeated-visit handling and lawful private evaluation access. Qualified
labelers need an explicit eight-target protocol, including H systemic-versus-
retinal meaning, O membership and N rules, with known positive/negative/unknown
states and blinded disagreement handling. Model predictions must not choose cases
or labels. Keep annotation source and uncertainty; do not manufacture compatibility
if ODIR's own semantics remain unresolved.

Plan sample size using prevalence, clustering, expected paired-model dependence
and the intended precision before outcome analysis. As a feasibility illustration,
100 independent positive components per class gives a worst-case binomial 95%
recall half-width near 0.10 before clustering/multiplicity inflation; it is **not**
a clinical recall target, power guarantee or sufficient full study design. Negative
support also matters, especially for H/O false positives. A formal precision/power
analysis is pending; rare H may govern feasibility. Current H errors are not
explained solely by its 67 positives. If rare-class enrichment is used, preserve
sampling probabilities and appropriate population weights; unweighted precision,
F1 or calibration from an enriched sample cannot represent population performance.

No all-eight cohort, licensed acquisition, annotation service or external
inference is authorized/started by this plan. Do not pay for access, submit access
forms, upload restricted data or contact others automatically. Existing Kaggle
credentials do not cover PhysioNet. BRSET should only enter a cloud/Kaggle workflow
if its DUA and the approved storage/processing arrangement permit it; otherwise
use an authorized private local/institutional CPU route, without paid hosting.
