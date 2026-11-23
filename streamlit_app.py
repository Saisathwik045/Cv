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
from dashboard.presentation import LABELS, SIGNALS, label_name, render_methodology

st.set_page_config(page_title='TrustRetina · Research', page_icon='◉', layout='wide')
st.markdown('''<style>
.block-container {max-width:1160px;padding-top:4.4rem} h1 {letter-spacing:-.04em}
[data-testid="stMetric"] {background:#eef5f4;border:1px solid #d4e6e1;border-radius:12px;padding:16px}
[data-testid="stSidebar"] {background:#eef5f4} .stButton>button {border-radius:8px}
[data-testid="stImage"] img {max-width:100%;height:auto}
@media print {
@page {size:A4;margin:12mm}
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"],
[data-testid="stMain"],[data-testid="stMainBlockContainer"] {height:auto!important;overflow:visible!important;position:static!important}
[data-testid="stSidebar"],[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="stFileUploader"],[data-testid="stButton"],[data-testid="stDownloadButton"] {display:none!important}
.block-container {padding:0!important;max-width:100%!important}
h1 {font-size:24pt!important} h2,h3 {break-after:avoid}
[data-testid="stHorizontalBlock"] {flex-wrap:wrap!important;gap:12px!important}
[data-testid="stColumn"] {min-width:0!important;flex:1 1 45%!important}
[data-testid="stImage"],table,tr {break-inside:avoid}
}
</style>''', unsafe_allow_html=True)
st.sidebar.title('◉ TrustRetina')
st.sidebar.caption('Frozen ConvNeXt-Tiny · Research demo')
page = st.sidebar.radio('Workspace', ['Paired-eye prediction', 'Research results'])
st.sidebar.divider()
st.sidebar.caption('Research use only. Predictions need human interpretation.')
st.sidebar.caption('No uploaded image storage or patient data in logs. Reset clears this session’s inputs and results; close the tab to end the session.')
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
    render_methodology()
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
    st.subheader('Per-label held-out performance')
    st.caption('Recorded 350-patient, patient-equal results at the fixed threshold. Support is the number of positive reference patients, not the number of correct predictions.')
    per_class = final['results']['metrics']['calibrated']['patient_equal']['classification']['per_class']
    st.dataframe(pd.DataFrame([{'Label': label_name(k), 'Precision': per_class[k]['precision'],
                               'Recall': per_class[k]['recall'], 'F1': per_class[k]['f1'],
                               'Positive support': per_class[k]['positive'],
                               'Negative support': per_class[k]['negative']} for k in final['class_order']]),
                 hide_index=True, width='stretch')
    with st.expander('How to interpret evaluation metrics'):
        st.write('Precision: among predicted positives, how many reference labels are positive. Recall: among reference positives, how many are detected. F1 balances precision and recall. Macro averages treat each label equally; micro averages pool label decisions. AUROC and average precision describe rankings, not a guarantee for one patient. Rare-label support limits precision of estimates; see the recorded report for intervals and undefined-metric flags.')
    st.subheader('Recorded evaluation · confusion matrix')
    st.caption('Historical 350-patient evaluation, not the current uploaded pair. Each label has its own binary matrix; this is a multilabel model.')
    from dashboard.charts import confusion_rows, confusion_chart
    label = st.selectbox('Evaluation label', final['class_order'], key='evaluation_matrix_label', format_func=label_name)
    counts = final['results']['metrics']['calibrated']['patient_equal']['classification']['per_class'][label]
    st.altair_chart(confusion_chart(confusion_rows(counts)), use_container_width=True)
    st.caption('Counts use patient-equal weighting and recorded patient-derived reference labels. Rows: reference absent/present; columns: predicted absent/present. TN/FP are the top row; FN/TP are the bottom row. These are research labels, not verified eye diagnoses.')
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
    st.caption('Use original JPG/PNG files from the same patient. Left/right refers to the patient’s eye, not its position on your screen. Filename checks cannot verify anatomy; confirm the source laterality before analysing.')
    left_column, right_column = st.columns(2)
    generation = st.session_state.get('upload_generation', 0)
    left = left_column.file_uploader('Left eye', type=['jpg', 'jpeg', 'png'], key=f'left_{generation}')
    right = right_column.file_uploader('Right eye', type=['jpg', 'jpeg', 'png'], key=f'right_{generation}')
    contents = [x.getvalue() if x is not None else None for x in (left, right)]
    signature = reconcile_uploads(st.session_state, *contents)
    from dashboard.uploads import reconcile_confirmation
    reconcile_confirmation(st.session_state, signature)
    valid = True
    decoded = []
    for col, data, label in zip((left_column, right_column), contents, ('Left eye', 'Right eye')):
        if data is not None:
            try:
                from trustretina.inference.service import decode_image
                image = decode_image(data)
                decoded.append(image)
                col.image(image, caption=label, use_container_width=True)
            except ImportError:
                col.error('CPU dependencies are unavailable. Use Python 3.12 and requirements/app-cpu.lock, then rebuild.')
                decoded.append(None)
                valid = False
            except ValueError as error:
                col.error(str(error))
                decoded.append(None)
                valid = False
        else:
            decoded.append(None)
    from dashboard.uploads import pair_errors
    issues = pair_errors(getattr(left, 'name', ''), getattr(right, 'name', ''), contents, decoded)
    for issue in issues:
        st.error(issue)
    valid = valid and not issues
    confirmed = st.checkbox('I confirm these are the left and right eyes of the same patient.', key='pair_confirmed', disabled=None in contents or not valid)
    if not valid or not confirmed:
        for key in ('prediction', 'explanation', 'explanation_target'):
            st.session_state.pop(key, None)
    if None in contents:
        st.info('Both eyes are required for the evaluated patient-level predictor. A single image is never duplicated to create a pair.')
    if st.button('Analyze paired images', type='primary', disabled=None in contents or not valid or not confirmed):
        try:
            from trustretina.inference.bundle import DEFAULT_BUNDLE
            service = get_service(os.environ.get('TRUSTRETINA_BUNDLE', str(DEFAULT_BUNDLE)))
            with st.spinner('Preprocessing both eyes, predicting eight labels and running 20 trained-head passes…'):
                st.session_state['prediction'] = service.predict(*contents)
            st.session_state.pop('explanation', None)
        except ImportError:
            st.error('CPU dependencies are unavailable. Use Python 3.12 and requirements/app-cpu.lock, then rebuild.')
        except (ValueError, OSError, RuntimeError) as error:
            st.error(f'Prediction unavailable: {error}')
    result = st.session_state.get('prediction')
    if result:
        st.subheader('All predicted-positive labels')
        st.write('Labels above threshold: ' + (', '.join(label_name(k) for k in result['positive_labels']) or 'None at the fixed p ≥ 0.5 rule. This does not establish a healthy retina.'))
        st.caption('N: normal label · D: diabetes · G: glaucoma · C: cataract · A: age-related macular degeneration · H: hypertension · M: pathological myopia · O: other abnormalities. Patient-derived research labels are not verified eye diagnoses.')
        probabilities = pd.DataFrame({'Label': result['class_order'], 'Raw probability': list(result['raw_probabilities'].values()), 'Calibrated probability': list(result['calibrated_probabilities'].values())})
        probabilities.insert(1, 'Research label', [LABELS[k] for k in result['class_order']])
        st.dataframe(probabilities, hide_index=True, width='stretch', column_config={'Raw probability': st.column_config.ProgressColumn(min_value=0, max_value=1, format='%.4f'), 'Calibrated probability': st.column_config.ProgressColumn(min_value=0, max_value=1, format='%.4f')})
        st.caption('Both-eye sigmoid outputs are averaged first; the saved shared temperature acts on the logit of that mean. The fixed 0.5 rule preserves the same raw/calibrated positive labels.')
        st.subheader('Prediction profile')
        from dashboard.charts import probability_chart
        st.altair_chart(probability_chart(result), use_container_width=True)
        st.caption('Light bars: raw probabilities. Dark bars: calibrated probabilities. Dashed line: fixed 0.5 positive-label threshold. The eight independent probabilities need not sum to 100%; they are not accuracy scores. A confusion matrix for this uploaded pair requires verified reference labels; see Research results for the recorded evaluation matrices.')
        a, b, c = st.columns(3)
        sig = result['signals']['calibrated']
        a.metric('Confidence score', f"{sig['confidence']:.4f}", help=SIGNALS['confidence'])
        b.metric('Head-only variance', f"{sig['uncertainty']:.6f}", help=SIGNALS['uncertainty'])
        c.metric('Technical quality proxy', f"{sig['quality']:.4f}", help=SIGNALS['quality'])
        with st.expander('What do these signals mean?'):
            for name, detail in SIGNALS.items():
                st.write(f'**{name.title()}** — {detail}')
        st.caption('Confidence = minimum certainty across the eight labels. Uncertainty = maximum label variance across 20 trained-head dropout passes; backbone and normalization stay in evaluation mode. Quality = frozen training-feature ranks, taking the lower eye score. These are experimental signals, not reliability percentages or clinical quality grades.')
        st.caption(f"{result['device']} · paired inference {result['inference_seconds']:.3f}s · frozen ConvNeXt-Tiny epoch 2")
        st.download_button('Download research result JSON', json.dumps(result, indent=2, allow_nan=False), file_name='trustretina-research-result.json', mime='application/json')
        st.divider()
        st.subheader('Explain one target · both eyes')
        target = st.selectbox('Explanation target (separate from the full positive-label list)', result['class_order'], index=max(range(8), key=lambda i: list(result['raw_probabilities'].values())[i]), format_func=label_name)
        if st.button('Generate per-eye Grad-CAM'):
            try:
                from trustretina.inference.bundle import DEFAULT_BUNDLE
                with st.spinner('Computing target gradients without changing model parameters…'):
                    service = get_service(os.environ.get('TRUSTRETINA_BUNDLE', str(DEFAULT_BUNDLE)))
                    cam, overlays = service.explain(*contents, result['class_order'].index(target))
                    st.session_state['explanation'] = (overlays, cam['zero_maps'])
                    st.session_state['explanation_target'] = target
            except ImportError:
                st.error('CPU dependencies are unavailable. Rebuild the Python 3.12 CPU environment.')
            except (ValueError, OSError, RuntimeError) as error:
                st.error(f'Explanation unavailable: {error}')
        if st.session_state.get('explanation_target') == target and st.session_state.get('explanation'):
            overlays, zeros = st.session_state['explanation']
            cols = st.columns(2)
            for col, image, eye, zero in zip(cols, overlays, ('Left', 'Right'), zeros):
                col.image(image, caption=f'{eye} eye · {label_name(target)}' + (' · zero activation map' if zero else ''), use_container_width=True)
            st.caption('Grad-CAM explains the selected raw mean-eye probability on the 224-pixel letterbox canvas. A heatmap is not verified lesion localization and does not validate the prediction.')
        st.divider()
        st.subheader('Complete demonstration report')
        st.caption('Includes both input previews, full prediction table, probability chart, signal explanations, current-target Grad-CAM, model details and the prediction record. Export is generated in memory. It contains your images: keep the downloaded copy private; reset cannot erase it.')
        current_maps = st.session_state.get('explanation_target') == target and st.session_state.get('explanation')
        if current_maps:
            try:
                from dashboard.report import demonstration_report
                from trustretina.inference.service import decode_image
                report_html = demonstration_report(result, [decode_image(x) for x in contents],
                                                    current_maps[0], target, zero_maps=current_maps[1])
                st.download_button('Download complete demonstration report', report_html,
                                   file_name='trustretina-demonstration-report.html', mime='text/html')
                st.caption('Open the downloaded HTML file locally, then choose Print → Save as PDF. The self-contained report has no external scripts or image links and includes the full content, not just the visible screen.')
            except (ImportError, ValueError, OSError):
                st.error('The complete report could not be prepared. Your prediction JSON remains available; no partial report is labelled complete.')
        else:
            st.info('Generate per-eye Grad-CAM for the selected target to enable the complete report download.')
