# Final submission checklist and reproducibility guide

**Development research submission; independent validation incomplete.**
The [final report](FINAL_RESEARCH_REPORT.md), [phase audit](PHASE_COMPLETION_AUDIT.md)
and [presentation outline](FINAL_PRESENTATION_OUTLINE.md) form the submission pack.
The original demo, models, thresholds and all calibration results remain frozen.

## Public submission contents

- [Examiner guide](EXAMINER_GUIDE.md): proposed rubric, timed evidence walkthrough,
  failure rehearsal and remaining human sign-off; dashboard **Examiner rehearsal**
  provides reasoning feedback without models, patient data or awarded grades.

- [Research defence chapter](ACADEMIC_RESEARCH_DEFENCE.md): research questions,
  decision trail, failure analysis, contributions, reproduction and examiner walkthrough.
- [All-eight-class comparison](ALL_EIGHT_CLASS_COMPARISON.md): recorded class
  errors, support and ECE; no independent winner or new outcome computation.
- Dashboard **Research study & viva** page; distinguish its saved V14/V15 evidence
  from the original frozen live predictor. Complete the authorship/assistance table.

- Report: abstract, architecture, data/label semantics, leakage control,
  preprocessing/training, five-model/later studies, V15, limitations and references.
- Completed phases through V15 audited against linked public evidence.
- Existing aggregate tables/figures and
  [public source digest index](../results/submission-audit/source-provenance.json).
- Presentation outline, synthetic demonstration run sheet and current test/CI evidence.

Exclude patient records/images/labels/predictions, credentials, original manifests,
private inventories/approvals, fitted private artifact files and checkpoints.
Public aggregate reporting does not authorize distributing underlying records.

## Linux CPU software reproduction

Use Python3.11 or3.12. From a clean checkout on a supported Linux CPU machine:

```sh
python3.12 -m venv .venv-submission
. .venv-submission/bin/activate
python -m pip install -r requirements/full-v5-reports.lock -r requirements/app-cpu.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m unittest discover -s tests -v
python -m compileall -q src scripts tests streamlit_app.py
trustretina check-config --config configs/phase0.toml
trustretina plan --config configs/phase0.toml
trustretina check-config --config configs/phase1.toml
trustretina check-development --config configs/development-v3.toml
trustretina check-contained-development --config configs/development-v4.toml
git diff --check
```

Locks pin versions, not wheel-file supply-chain hashes. Linux `+cpu` Torch wheels
are not a Mac recipe. Package metadata accepting later Python does not establish
support beyond the3.11/3.12 check matrix. Fresh installs are exercised by CI; do
not infer native Mac installation from successful Linux checks.

Enable the optional photographic tests in the test environment:

```sh
python -m pip install -e '.[duplicate-review]'
python -m pip check
python -m unittest discover -s tests -p test_photo_verification.py -v
python -m unittest discover -s tests -p test_bounded_review.py -v
```

Recovery remains separate from app runtime. CI installs the unchanged recovery
lock and runs sterile SDK/public-plan checks, not private recovery or account
activation. Unit optimizer/model fixtures are synthetic, not research-model
training or repeated real calibration. Existing app tests may display public
historical aggregate evidence; no private final-cohort records or replay are used.

## Current synthetic Streamlit review

```sh
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_demo.py -v
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_submission_demo.py -v
python -m unittest discover -s tests -p test_mac_launcher.py -v
bash -n Start_TrustRetina.command Setup_TrustRetina.command scripts/macos_bootstrap.sh
```

Submission tests generate PNGs and replace the service with a fake. They check
missing/invalid-eye blocking, eight-class presentation, REVIEW REQUIRED, download
control presence, explanation-target selection, reset and visible bundle failure.
They do not load private weights, establish clinical accuracy, verify an external
browser download or Cloud preview, or prove native Mac runtime behavior. The app
entry point and inference code are unchanged.

For a separately permitted real local demo, use the exact original inference-v8
bundle and [installation guide](DEMO_V8.md), then `python scripts/run_demo.py` on
the same machine as the browser. This needs the verified private bundle; recovery
and real image inference are outside this submission task. The original demo does
not silently load V14 checkpoints or V15 temperatures. Missing bundles fail closed.

## Mac scope

Use current `main`, macOS13+, native Apple-silicon Python3.11/3.12 and the separate
Mac lock/launcher in [Mac quickstart](MACOS_QUICKSTART.md). Setup requires a
previously approved bundle or separately approved exact recovery. The server runs
without Kaggle variables after setup. Portable Linux/wheel checks do not verify
Finder/Gatekeeper, native install, checkpoint loading/latency or numerical parity.
A real Mac hands-on check remains outstanding and must not be represented as passed.

## Readiness and author actions

| Item | Status / next action |
|---|---|
| Development through V15 | Complete in recorded evidence; retain positive and negative findings |
| Current software checks | See STATUS/PR for observed local/CI results; synthetic tests are not new science |
| Original frozen demo | Preserved; real use needs separately permitted bundle/images |
| Final report/presentation | Prepared; student adds author/institution/supervisor, institutional PDF/slides formatting, citation review and sign-off |
| Independent confirmation | Incomplete; preregister untouched cohort/endpoint, authoritative compatible all-eight labels, overlap/role checks, support and precision/power plan |
| External access | BRSET credentialing/CITI/DUA and actual compatibility unresolved; no automatic applications/downloads |
| Clinical acceptance | Unsupported; REVIEW REQUIRED, no fitted acceptance policy or clinical safety claim |
| Native Mac execution | Unverified on this Linux host; separately test the supported Mac setup |
| Durable private preservation | Unverified; destination/recipient/transfer approval, encrypted backup, authenticated full-hash restore drill and second copy outstanding |
| PR merge/release | Reviewer/user decision; never automatically merged |

**Submission readiness:** prepared for academic review as development research,
conditional on passing current checks, reviewer approval, institutional formatting
and author/supervisor sign-off. It is not independently clinically validated.
A development-only report can be submitted with these limitations; independent
study execution and private backup remain separate authorizations.
