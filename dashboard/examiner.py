"""Public-only academic rehearsal, not grading or research evaluation."""
import json

SCENARIOS = (
    {'id': 'labels', 'question': 'Two disease targets can be positive together. Which output contract is appropriate?',
     'options': ('Eight sigmoid probabilities with multi-label targets', 'Softmax followed by one argmax diagnosis'), 'correct': 0,
     'reason': 'ODIR targets are patient-derived and multi-label. An eye inherits these targets; it does not acquire an independently verified eye diagnosis.',
     'evidence': 'docs/ACADEMIC_RESEARCH_DEFENCE.md'},
    {'id': 'leakage', 'question': 'The left eye is in training and the right eye in validation. Is this acceptable?',
     'options': ('Yes, because the image files differ', 'No; group both eyes and containment components before splitting'), 'correct': 1,
     'reason': 'Patient/component overlap can leak identity and related information. Grouping is necessary; it does not undo repeated validation use or prove all duplicates were detected.',
     'evidence': 'docs/EXPERIMENT_PROTOCOL.md'},
    {'id': 'calibration', 'question': 'V15 NLL decreases but ten-bin ECE increases. What should the report conclude?',
     'options': ('The model is uniformly more reliable', 'Retain both findings; the scalar objective does not guarantee lower ECE'), 'correct': 1,
     'reason': 'V15 has slight NLL/Brier decreases and worse ECE in all four cohort/resolution contexts. Calibration diagnostics are in sample; reused validation is descriptive.',
     'evidence': 'results/calibration-v15/RESULTS.md'},
    {'id': 'threshold', 'question': 'For positive T, does sigmoid(logit(p)/T) change a fixed p >= 0.5 decision?',
     'options': ('No; the sign and 0.5 boundary are preserved', 'Yes; lower NLL necessarily fixes false negatives'), 'correct': 0,
     'reason': 'A positive scalar preserves the boundary after mean-eye aggregation. Classification metrics remain unchanged here; calibration alone does not establish better discrimination.',
     'evidence': 'docs/ALL_EIGHT_CLASS_COMPARISON.md'},
    {'id': 'explanation', 'question': 'A clear Grad-CAM map and low head-only MC variance are displayed. What do they prove?',
     'options': ('Clinical safety and correct lesion localization', 'Neither; sensitivity and head variability are limited experimental signals'), 'correct': 1,
     'reason': 'There is no lesion-ground-truth verification. Head-only MC omits backbone uncertainty, and technical quality is not clinical gradability. REVIEW REQUIRED remains mandatory.',
     'evidence': 'docs/ACADEMIC_RESEARCH_DEFENCE.md'},
    {'id': 'confirmation', 'question': 'What would provide stronger confirmation for V14/V15 after repeated development use?',
     'options': ('Preregister suitable untouched data, compatible labels, overlap controls and endpoints', 'Retest the historical final cohort and call it independent'), 'correct': 0,
     'reason': 'The historical final cohort is already used. A compatible independent cohort, support/precision plan and separately authorized evaluation are required; no such study runs in this rehearsal.',
     'evidence': 'docs/FINAL_SUBMISSION_CHECKLIST.md'},
)


def feedback(answers):
    rows = []
    for item in SCENARIOS:
        answer = answers.get(item['id'])
        valid = type(answer) is int and 0 <= answer < len(item['options'])
        rows.append({'topic': item['id'], 'status': 'UNANSWERED' if not valid else
                     ('ALIGNED' if answer == item['correct'] else 'REVISIT'),
                     'explanation': item['reason'], 'evidence': item['evidence']})
    return rows


def rehearsal_export(answers):
    return {'schema': 'academic_rehearsal_v1', 'purpose': 'Self-practice only; not an examiner grade',
            'research_evaluation': False, 'clinical_validation': False,
            'decision': 'REVIEW REQUIRED', 'topics': feedback(answers)}


def render_examiner_page():
    import streamlit as st
    st.title('Examiner rehearsal')
    st.warning('Academic self-practice only · REVIEW REQUIRED · Not a grade, clinical validation or proof of authorship.')
    st.write('A teacher should expect a justified design, leakage controls, reproducible evidence, honest negative findings and clear ownership. Explain each answer aloud before checking it. This page uses public teaching scenarios; it loads no model or patient records and computes no research metrics.')
    st.info('Use Research study & viva for verified recorded results, Guided synthetic demo for software operation, and the examiner guide for evidence and submission requirements.')
    if st.button('Reset rehearsal'):
        for key in list(st.session_state):
            if key.startswith('examiner_'):
                st.session_state.pop(key)
        st.rerun()
    answers = {}
    for item in SCENARIOS:
        choice = st.radio(item['question'], item['options'], index=None, key='examiner_' + item['id'])
        if choice is not None:
            answers[item['id']] = item['options'].index(choice)
    if st.button('Check reasoning'):
        st.session_state['examiner_checked'] = True
    if st.session_state.get('examiner_checked'):
        st.caption('Feedback updates with your current choices. ALIGNED means agreement with this teaching key, not demonstrated mastery. Defend the reasoning and inspect the linked repository evidence.')
        for item, row in zip(SCENARIOS, feedback(answers)):
            with st.expander(row['topic'] + ' · ' + row['status'], expanded=True):
                st.write(row['explanation'])
                st.caption('Repository evidence: ' + row['evidence'])
        st.download_button('Download rehearsal feedback', json.dumps(rehearsal_export(answers), indent=2),
                           file_name='trustretina-rehearsal.json', mime='application/json')
    from pathlib import Path
    guide = Path(__file__).resolve().parents[1] / 'docs/EXAMINER_GUIDE.md'
    if guide.is_file():
        st.download_button('Download examiner guide', guide.read_text(),
                           file_name='trustretina-examiner-guide.md', mime='text/markdown')
    st.write('Before submission: complete author/institution/supervisor details, citations, actual contribution and assistance disclosures, institutional formatting and supervisor sign-off. Rehearse cloud access; runtime availability is not guaranteed. No automatic pass or project-completion certificate is issued.')
