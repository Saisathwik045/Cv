"""Plain-language presentation of the unchanged research pipeline."""
LABELS = {
    'N': 'Normal label', 'D': 'Diabetes', 'G': 'Glaucoma', 'C': 'Cataract',
    'A': 'Age-related macular degeneration', 'H': 'Hypertension',
    'M': 'Pathological myopia', 'O': 'Other abnormalities',
}
SIGNALS = {
    'confidence': 'Minimum certainty across the eight labels: min(max(p, 1−p)). A label near 0.5 lowers this score. It is not the probability that the prediction is correct.',
    'uncertainty': 'Largest per-label variance across 20 trained-head dropout passes. Lower variance means less variation in those passes, not proven diagnostic reliability; backbone uncertainty is not measured.',
    'quality': 'Technical training-feature rank, taking the lower of the two eye scores. It is not a clinical image-quality grade, and a larger value does not guarantee a correct prediction.',
}
WORKFLOW = 'Image pair → EXIF/RGB + 224 px letterbox + normalization → frozen ConvNeXt-Tiny → mean-eye sigmoid probabilities → saved shared temperature → confidence → selected-target Grad-CAM. Predictions require human interpretation.'


def label_name(code):
    return f'{LABELS[code]} ({code})'


def render_methodology():
    import streamlit as st
    st.subheader('How the project works')
    st.write(WORKFLOW)
    with st.expander('Dataset, splitting and model selection'):
        st.write('Source: ODIR-5K. The original data.xlsx supplies eight independent patient-derived research targets, inherited by both eye images. They are not verified diagnoses for each eye. Several labels can be positive together.')
        st.write('The historical development records describe 3,500 patients / 7,000 images: 2,275 training, 525 validation, 350 calibration and 350 held-out evaluation patients. Both eyes and containment components stay together across partitions to reduce leakage. Duplicate/containment limitations remain documented.')
        st.write('Five backbones were compared. Validation loss selected each checkpoint; patient validation macro AUROC selected ConvNeXt-Tiny. The deployed checkpoint is epoch 2 at 224 px. A separate calibration role fitted one shared temperature after mean-eye aggregation; the positive-label threshold stays at 0.5.')
        st.write('The 350-patient held-out evaluation was executed once with frozen parameters and is now historical evidence. It is not reused to choose new models, thresholds or policies. No clinically validated automatic acceptance policy or independent clinical validation is established.')
        st.markdown('[Full research report and provenance](https://github.com/Saisathwik045/Cv/blob/main/docs/FINAL_RESEARCH_REPORT.md)')
