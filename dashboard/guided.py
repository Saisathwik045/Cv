"""Generated-image operational walkthrough; original CPU inference only on click."""
import io
import json

CLASSES = list('NDGCAHMO')


def synthetic_pair(replacement=False):
    from PIL import Image, ImageDraw
    images = []
    for color in (((20, 90, 70) if replacement else (100, 35, 20)), (65, 35, 100)):
        image = Image.new('RGB', (320, 240), 'black')
        draw = ImageDraw.Draw(image)
        draw.ellipse((35, 10, 285, 230), fill=color)
        draw.line((50, 190, 260, 40), fill=(220, 140, 50), width=6)
        draw.text((20, 110), 'SYNTHETIC NON-PATIENT', fill='white')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        images.append(buffer.getvalue())
    return tuple(images)


def clear_output(state):
    for name in ('guided_result', 'guided_cam', 'guided_cam_target', 'guided_target'):
        state.pop(name, None)


def clear_demo(state):
    for key in list(state):
        if key.startswith('guided_'):
            state.pop(key)


def _load_service(bundle):
    from trustretina.inference.service import InferenceService
    return InferenceService(bundle, device='cpu')


def predict_and_explain(service, pair):
    import math
    from trustretina.inference.bundle import verified_contract
    contract = verified_contract()
    result = service.predict(*pair)
    if result['device'] != 'cpu' or result['class_order'] != CLASSES or result['decision'] != 'REVIEW REQUIRED':
        raise ValueError('Unexpected frozen demo output contract')
    for name in ('raw_probabilities', 'calibrated_probabilities'):
        if list(result[name]) != CLASSES or not all(math.isfinite(x) and 0 <= x <= 1 for x in result[name].values()):
            raise ValueError('Invalid demonstration probabilities')
    if result['mc_passes'] != 20 or result['threshold'] != .5:
        raise ValueError('Unexpected frozen reliability contract')
    if (result['best_epoch'], result['temperature'], result['checkpoint_sha256'], result['preprocessing_sha256']) != (
            2, contract['temperature'], contract['checkpoint_sha256'], contract['preprocessing_sha256']):
        raise ValueError('Original demo identity required; no V14/V15 substitution')
    if not all(math.isfinite(result['signals']['calibrated'][name]) for name in ('confidence', 'uncertainty', 'quality')):
        raise ValueError('Invalid demonstration signals')
    _, overlays = service.explain(*pair, 1)
    if len(overlays) != 2:
        raise ValueError('Both synthetic explanations are required')
    return result, overlays


def envelope(result):
    return {'schema': 'synthetic_demo_v1', 'input_kind': 'generated_geometric_non_patient',
            'interpretation': 'Operational smoke test only; no disease ground truth or medical meaning',
            'research_evaluation': False, 'prediction': result}


def render_guided_page():
    import os
    import pandas as pd
    import streamlit as st
    from trustretina.inference.bundle import DEFAULT_BUNDLE
    bundle = os.environ.get('TRUSTRETINA_BUNDLE', str(DEFAULT_BUNDLE))
    service = st.cache_resource(show_spinner=False)(_load_service)
    st.title('Guided synthetic demonstration')
    st.warning('SYNTHETIC NON-PATIENT INPUTS · REVIEW REQUIRED · No medical meaning or disease ground truth.')
    st.write('One click runs the original epoch-2, 224 px frozen model on generated geometric PNGs using CPU, then explains target D for both images. This tests software operation; it is not a research evaluation, clinical prediction or retinal performance test.')
    st.caption('No uploads, patient records, downloads of weights or fitting. Images/results stay in this session unless you explicitly export the demo JSON. The recorded research results remain in their separate workspaces.')
    run, replace, reset = st.columns(3)
    run_now = run.button('Run guided CPU demo', type='primary')
    if replace.button('Replace synthetic left image'):
        replacement = not st.session_state.get('guided_replacement', False)
        clear_output(st.session_state)
        st.session_state['guided_replacement'] = replacement
        st.rerun()
    if reset.button('Reset synthetic demonstration'):
        clear_demo(st.session_state)
        st.rerun()
    pair = synthetic_pair(st.session_state.get('guided_replacement', False))
    for column, data, eye in zip(st.columns(2), pair, ('Left', 'Right')):
        column.image(data, caption=eye + ' · generated geometric fixture, not a retinal photograph', width=320)
    if run_now:
        clear_output(st.session_state)
        try:
            with st.spinner('Verifying the original bundle, loading CPU weights, predicting once and explaining D…'):
                result, overlays = predict_and_explain(service(bundle), pair)
            st.session_state['guided_result'] = result
            st.session_state['guided_cam'] = overlays
            st.session_state['guided_cam_target'] = 'D'
        except (KeyError, TypeError, ValueError, OSError, RuntimeError):
            st.error('The original frozen CPU demonstration could not complete. No substitute model or invented result is displayed. Check Cloud readiness; no artifacts are downloaded automatically.')
    result = st.session_state.get('guided_result')
    if not result:
        st.info('Ready for step 1: run the generated pair. Replace invalidates old outputs; reset clears only this guided session and returns to the original generated pair.')
        return
    st.success('Operational prediction and initial both-image explanation completed. This is not evidence of medical validity.')
    st.subheader('Inspect all eight outputs · synthetic inputs only')
    st.dataframe(pd.DataFrame({'Class': CLASSES,
                              'Raw probability': [result['raw_probabilities'][k] for k in CLASSES],
                              'Calibrated probability': [result['calibrated_probabilities'][k] for k in CLASSES]}),
                 hide_index=True, width='stretch',
                 column_config={k: st.column_config.NumberColumn(format='%.6f') for k in ('Raw probability', 'Calibrated probability')})
    sig = result['signals']['calibrated']
    for col, label, key in zip(st.columns(3), ('Experimental confidence', 'Head-only MC variance', 'Technical quality proxy'), ('confidence', 'uncertainty', 'quality')):
        col.metric(label, f'{sig[key]:.6f}')
    st.caption('20 trained-head passes. These signals are not clinical confidence percentages, diagnostic uncertainty bounds or gradability labels. No positive-label interpretation is medically meaningful on generated shapes.')
    st.download_button('Download synthetic demo JSON', json.dumps(envelope(result), indent=2, allow_nan=False),
                       file_name='trustretina-synthetic-demo.json', mime='application/json')
    st.subheader('Explain one synthetic target')
    target = st.selectbox('Synthetic explanation target', CLASSES, index=1, key='guided_target')
    if st.button('Update synthetic explanation'):
        st.session_state.pop('guided_cam', None)
        try:
            with st.spinner('Computing the selected target explanation on CPU…'):
                _, overlays = service(bundle).explain(*pair, CLASSES.index(target))
            st.session_state['guided_cam'] = overlays
            st.session_state['guided_cam_target'] = target
        except (KeyError, TypeError, ValueError, OSError, RuntimeError):
            st.error('Synthetic explanation unavailable. No substitute heatmap is displayed.')
    if st.session_state.get('guided_cam_target') == target and st.session_state.get('guided_cam') is not None:
        for col, overlay, eye in zip(st.columns(2), st.session_state['guided_cam'], ('Left', 'Right')):
            col.image(overlay, caption=f'{eye} synthetic input · {target}', width=320)
    else:
        st.info('Update the explanation to display the selected target; an old target heatmap is not relabelled.')
    st.caption('Grad-CAM illustrates target sensitivity, not lesion localization or prediction correctness. Replace the left fixture to check stale-output clearing, then reset. The sidebar reset clears the entire app session.')
