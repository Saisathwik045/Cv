"""Local/private research demo. Launch: python -m streamlit run streamlit_app.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
import json
import os
import pandas as pd
import streamlit as st
from trustretina.app.state import reset_streamlit_session, reconcile_uploads
from trustretina.app.results import load_evidence

st.set_page_config(page_title='TrustRetina · Research', page_icon='◉', layout='wide')
st.markdown('''<style>
.block-container {max-width:1160px;padding-top:4.4rem} h1 {letter-spacing:-.04em}
[data-testid="stMetric"] {background:#eef5f4;border:1px solid #d4e6e1;border-radius:12px;padding:16px}
[data-testid="stSidebar"] {background:#eef5f4} .stButton>button {border-radius:8px}
</style>''', unsafe_allow_html=True)
st.sidebar.title('◉ TrustRetina')
st.sidebar.caption('Frozen ConvNeXt-Tiny · Research demo')
page = st.sidebar.radio('Workspace', ['Paired-eye prediction', 'Research results'])
st.sidebar.divider()
st.sidebar.write('**REVIEW REQUIRED** for every prediction.')
st.sidebar.caption('No automatic acceptance policy. No uploaded image storage or patient data in logs. Reset clears this session’s inputs and results; close the tab to end the session.')
if st.sidebar.button('Reset patient session', width='stretch'):
    reset_streamlit_session(st.session_state)
    st.rerun()


@st.cache_resource(show_spinner='Verifying and loading the frozen model…')
def get_service(bundle):
    from trustretina.inference.service import InferenceService
    return InferenceService(bundle)


def results_page():
    st.title('Evidence, with its limits')
    st.write('Validation selected the backbone. The separate held-out evaluation tested the frozen candidate once; it did not select parameters or reliability strategies.')
    try:
        validation, final, report = load_evidence()
    except (OSError, ValueError) as error:
        st.error(str(error))
        return
    st.subheader('Five-backbone validation comparison')
    table = pd.DataFrame(validation)[['backbone', 'patient_macro_auroc', 'patient_macro_auprc', 'patient_macro_f1', 'patient_micro_f1', 'best_epoch']]
    for column in table.columns[1:]:
        table[column] = pd.to_numeric(table[column])
    table = table.rename(columns={'backbone': 'Backbone', 'patient_macro_auroc': 'Macro AUROC', 'patient_macro_auprc': 'Macro AP', 'patient_macro_f1': 'Macro F1', 'patient_micro_f1': 'Micro F1', 'best_epoch': 'Best epoch'})
    st.dataframe(table, hide_index=True, width='stretch', column_config={name: st.column_config.NumberColumn(format='%.4f') for name in ('Macro AUROC', 'Macro AP', 'Macro F1', 'Micro F1')})
    st.caption('525 validation patients. Selection: patient macro AUROC. Best checkpoint: validation loss. Labelwise binary accuracy is the fraction of correct label bits, not diagnostic accuracy.')
    st.subheader('Frozen held-out evaluation · 350 patients')
    st.metric('Patient macro AUROC', f"{final['primary_estimate']:.4f}")
    # The generated report contains actual per-class, pooled and paired interval tables.
    st.markdown(report.partition('\n')[2].split('## Provenance and next phase')[0])
    st.download_button('Download verified research report', report, file_name='trustretina-final-research-report.md', mime='text/markdown')
    for name in ('patient_equal-reliability-diagrams.png', 'patient_equal-hamming_risk.png'):
        st.image(str(Path(__file__).resolve().parent / 'results/final-v7' / name))


if page == 'Research results':
    results_page()
else:
    st.caption('RESEARCH DEMONSTRATION / PAIRED-EYE INFERENCE')
    st.title('Two eyes. One patient-level prediction.')
    st.write('Upload the left and right fundus images of the same patient. This frozen research model returns eight independent label probabilities, not a clinical diagnosis.')
    left_column, right_column = st.columns(2)
    generation = st.session_state.get('upload_generation', 0)
    left = left_column.file_uploader('Left eye', type=['jpg', 'jpeg', 'png'], key=f'left_{generation}')
    right = right_column.file_uploader('Right eye', type=['jpg', 'jpeg', 'png'], key=f'right_{generation}')
    contents = [x.getvalue() if x is not None else None for x in (left, right)]
    reconcile_uploads(st.session_state, *contents)
    valid = True
    for col, data, label in zip((left_column, right_column), contents, ('Left eye', 'Right eye')):
        if data is not None:
            try:
                from trustretina.inference.service import decode_image
                col.image(decode_image(data), caption=label, width=320)
            except ValueError as error:
                col.error(str(error))
                valid = False
    if None in contents:
        st.info('Both eyes are required for the evaluated patient-level predictor. A single image is never duplicated to create a pair.')
    if st.button('Analyze paired images', type='primary', disabled=None in contents or not valid):
        try:
            from trustretina.inference.bundle import DEFAULT_BUNDLE
            service = get_service(os.environ.get('TRUSTRETINA_BUNDLE', str(DEFAULT_BUNDLE)))
            with st.spinner('Preprocessing both eyes, predicting eight labels and running 20 trained-head passes…'):
                st.session_state['prediction'] = service.predict(*contents)
            st.session_state.pop('explanation', None)
        except (ValueError, OSError, RuntimeError) as error:
            st.error(f'Prediction unavailable: {error}')
    result = st.session_state.get('prediction')
    if result:
        st.warning('REVIEW REQUIRED · No validated automatic acceptance policy is fitted.')
        st.subheader('All predicted-positive labels')
        st.write(', '.join(result['positive_labels']) or 'None at the fixed p ≥ 0.5 rule. This does not establish a healthy retina.')
        st.caption('N: normal label · D: diabetes · G: glaucoma · C: cataract · A: age-related macular degeneration · H: hypertension · M: pathological myopia · O: other abnormalities. Patient-derived research labels are not verified eye diagnoses.')
        probabilities = pd.DataFrame({'Label': result['class_order'], 'Raw probability': list(result['raw_probabilities'].values()), 'Calibrated probability': list(result['calibrated_probabilities'].values())})
        st.dataframe(probabilities, hide_index=True, width='stretch', column_config={'Raw probability': st.column_config.ProgressColumn(min_value=0, max_value=1, format='%.4f'), 'Calibrated probability': st.column_config.ProgressColumn(min_value=0, max_value=1, format='%.4f')})
        st.caption('Both-eye sigmoid outputs are averaged first; the saved shared temperature acts on the logit of that mean. The fixed 0.5 rule preserves the same raw/calibrated positive labels.')
        a, b, c = st.columns(3)
        sig = result['signals']['calibrated']
        a.metric('Confidence score', f"{sig['confidence']:.4f}")
        b.metric('Head-only variance', f"{sig['uncertainty']:.6f}")
        c.metric('Technical quality proxy', f"{sig['quality']:.4f}")
        st.caption('Confidence = minimum certainty across the eight labels. Uncertainty = maximum label variance across 20 trained-head dropout passes; backbone and normalization stay in evaluation mode. Quality = frozen training-feature ranks, taking the lower eye score. These are experimental signals, not reliability percentages or clinical quality grades.')
        st.caption(f"{result['device']} · paired inference {result['inference_seconds']:.3f}s · frozen ConvNeXt-Tiny epoch 2")
        st.download_button('Download research result JSON', json.dumps(result, indent=2, allow_nan=False), file_name='trustretina-research-result.json', mime='application/json')
        st.divider()
        st.subheader('Explain one target · both eyes')
        target = st.selectbox('Explanation target (separate from the full positive-label list)', result['class_order'], index=max(range(8), key=lambda i: list(result['raw_probabilities'].values())[i]))
        if st.button('Generate per-eye Grad-CAM'):
            try:
                from trustretina.inference.bundle import DEFAULT_BUNDLE
                with st.spinner('Computing target gradients without changing model parameters…'):
                    service = get_service(os.environ.get('TRUSTRETINA_BUNDLE', str(DEFAULT_BUNDLE)))
                    cam, overlays = service.explain(*contents, result['class_order'].index(target))
                    st.session_state['explanation'] = (overlays, cam['zero_maps'])
                    st.session_state['explanation_target'] = target
            except (ValueError, OSError, RuntimeError) as error:
                st.error(f'Explanation unavailable: {error}')
        if st.session_state.get('explanation_target') == target and st.session_state.get('explanation'):
            overlays, zeros = st.session_state['explanation']
            cols = st.columns(2)
            for col, image, eye, zero in zip(cols, overlays, ('Left', 'Right'), zeros):
                col.image(image, caption=f'{eye} eye · {target}' + (' · zero activation map' if zero else ''), width=360)
            st.caption('Grad-CAM explains the selected raw mean-eye probability on the 224-pixel letterbox canvas. A heatmap is not verified lesion localization and does not validate the prediction.')
