"""Presentation of recorded public aggregates; no fitting, inference or private IO."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGGREGATE_SHA256 = 'f9f3ed43a53d53b8b77d6f415b5fc35406a1dc67cb8989f6009edbfb96c89469'
CLASSES = list('NDGCAHMO')

EVIDENCE_MAP = [
    {'Question': 'Which initial backbone?', 'Experiment': 'V5: five fixed backbones',
     'Finding': 'ConvNeXt-Tiny selected by validation macro AUROC',
     'Limit': 'Selection cohort is not independent confirmation'},
    {'Question': 'Do schedule or loss changes help?', 'Experiment': 'V9–V11: schedule, threshold CV, nested loss',
     'Finding': 'Tradeoffs retained; square-root weighting did not establish superiority',
     'Limit': 'V10 thresholds are exploratory and were not deployed'},
    {'Question': 'Does higher resolution help?', 'Experiment': 'V13 matched folds; V14 frozen 224/320 fits',
     'Finding': 'Aggregate ranking gains coexist with class-specific harms',
     'Limit': 'Conditional development evidence; no independent winner'},
    {'Question': 'Does a shared temperature improve reliability?', 'Experiment': 'V15: one scalar per frozen candidate',
     'Finding': 'Small NLL/Brier decreases, worse ECE, unchanged fixed-threshold decisions',
     'Limit': 'Calibration is in sample; validation was reused'},
    {'Question': 'Do uncertainty and quality rank errors better?', 'Experiment': 'V15: three fixed signal strategies',
     'Finding': 'Quality fusion worsened Hamming AURC',
     'Limit': 'Head-only MC; technical quality is not clinical gradability'},
]

VIVA = [
    ('What is your contribution?', 'An integrated paired-eye research system, experimental controls, reproducible comparisons and documented positive/negative findings. ConvNeXt, temperature scaling, MC Dropout and Grad-CAM are existing methods, not inventions of this project. The student must document their own implementation, analysis and external/AI assistance honestly.'),
    ('Why sigmoid rather than softmax?', 'The eight targets are multi-label. Several can be positive together; argmax would discard information. Patient-derived labels inherited by eyes are not independently verified eye diagnoses.'),
    ('How is leakage controlled?', 'Keep both eyes and containment components together across roles. Explain residual duplicate uncertainty and every reuse of validation; grouping does not erase model-selection exposure.'),
    ('Why can NLL improve while ECE worsens?', 'The scalar optimizes component-equal NLL, not ten-bin ECE or every class. Different objectives and sparse bins can disagree; report both outcomes.'),
    ('Why are classification metrics unchanged?', 'Positive temperature scaling after mean-eye aggregation preserves the fixed 0.5 boundary. It does not repair class errors or establish improved discrimination.'),
    ('Does a heatmap or low MC variance establish safety?', 'No. Grad-CAM is target sensitivity, not verified lesion localization. Trained-head variability omits backbone uncertainty and is not a clinical error guarantee.'),
    ('What would strengthen the science next?', 'Prospectively register a compatible untouched cohort, endpoints, overlap controls and support/power requirements. New inference/evaluation needs separate authorization; do not recycle the historical final cohort.'),
]


def load_academic_evidence(root=ROOT):
    base = Path(root) / 'results/calibration-v15'
    raw = (base / 'aggregate-results.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != AGGREGATE_SHA256:
        raise ValueError('Public V15 aggregate checksum mismatch')
    data = json.loads(raw)
    if data['class_order'] != CLASSES or data['deployment_default'] != 'REVIEW REQUIRED':
        raise ValueError('Public V15 presentation contract mismatch')
    for name, record in data['figures'].items():
        # The hash-bound JSON contains the reviewed relative figure inventory.
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Public V15 figure checksum mismatch')
    return data


def metric_rows(data):
    return [dict(Cohort=role, Resolution=run['resolution'], Mode=mode,
                 NLL=d[mode]['metrics']['patient_nll'],
                 Brier=d[mode]['metrics']['brier'],
                 ECE=d[mode]['metrics']['macro_ece_10_equal_width_bins'])
            for run in data['runs'].values() for role, d in run['roles'].items()
            for mode in ('raw', 'calibrated')]


def class_rows(data, role):
    result = []
    for label in CLASSES:
        for run in data['runs'].values():
            d = run['roles'][role]
            c = next(row for row in d['per_class_classification'] if row['label'] == label)
            raw = next(row for row in d['raw']['per_class_reliability'] if row['label'] == label)
            cal = next(row for row in d['calibrated']['per_class_reliability'] if row['label'] == label)
            result.append({'Class': label, 'Resolution': run['resolution'],
                           'Positive': c['positive'], 'Negative': c['negative'],
                           'Precision': c['precision'] if c['precision_defined'] else None,
                           'Recall': c['recall'] if c['recall_defined'] else None,
                           'F1': c['f1'] if c['f1_defined'] else None,
                           'FP': c['fp'], 'FN': c['fn'], 'Raw ECE': raw['ece'], 'Calibrated ECE': cal['ece']})
    return result


def risk_rows(data, role):
    return [{'Resolution': run['resolution'], 'Mode': mode, 'Strategy': strategy,
             'Hamming AURC': values['hamming_aurc'], 'Any-error AURC': values['any_error_aurc']}
            for run in data['runs'].values() for mode in ('raw', 'calibrated')
            for strategy, values in run['roles'][role][mode]['risk_coverage'].items()]


def render_academic_page():
    import pandas as pd
    import streamlit as st
    st.title('Research study & viva')
    st.warning('REVIEW REQUIRED · Development research; independent clinical validation is incomplete.')
    st.info('The live predictor remains the original v5 epoch-2 / Phase6 demo. V14/V15 below are separate recorded research candidates; this page runs no models or fitting.')
    try:
        data = load_academic_evidence()
    except (OSError, ValueError, KeyError):
        st.error('Verified public research evidence is unavailable. No substitute results are displayed.')
        return
    evidence, results, figures, critical, viva = st.tabs([
        'Questions & experiments', 'V15 & all eight classes', 'Reliability diagrams',
        'Critical analysis', 'Viva & reproduction'])
    with evidence:
        st.subheader('Question → experiment → finding → limitation')
        st.dataframe(pd.DataFrame(EVIDENCE_MAP), hide_index=True, width='stretch')
        st.write('Compare within each prespecified experiment. Different training sizes, cohorts, checkpoint-selection rules and development exposure cannot be pooled into one unbiased longitudinal leaderboard.')
    with results:
        st.subheader('Two shared temperatures, separate from the live demo')
        st.dataframe(pd.DataFrame([{'Run': name, 'Resolution': run['resolution'],
                                    'Shared temperature': run['fit']['temperature']}
                                   for name, run in data['runs'].items()]), hide_index=True, width='stretch',
                     column_config={'Shared temperature': st.column_config.NumberColumn(format='%.12f')})
        st.caption('Calibration: 350 patients / 331 components, in-sample fit diagnostics. Validation: 525 / 518, reused-development descriptions. NLL/Brier here use patient weighting; ECE is the mean eight-class ten-bin ECE.')
        st.dataframe(pd.DataFrame(metric_rows(data)), hide_index=True, width='stretch',
                     column_config={k: st.column_config.NumberColumn(format='%.6f') for k in ('NLL', 'Brier', 'ECE')})
        role = st.selectbox('Recorded cohort', ['validation', 'calibration'])
        st.subheader('All-eight-class comparison, including errors and support')
        st.caption('Precision/recall/F1 and FP/FN are unchanged by the shared temperature at p ≥ 0.5. Positive/negative columns count patients, not independent eyes. Undefined metrics remain missing.')
        st.dataframe(pd.DataFrame(class_rows(data, role)), hide_index=True, width='stretch',
                     column_config={k: st.column_config.NumberColumn(format='%.6f') for k in ('Precision', 'Recall', 'F1', 'Raw ECE', 'Calibrated ECE')})
        st.subheader('Saved uncertainty and quality strategies')
        st.dataframe(pd.DataFrame(risk_rows(data, role)), hide_index=True, width='stretch',
                     column_config={k: st.column_config.NumberColumn(format='%.6f') for k in ('Hamming AURC', 'Any-error AURC')})
        st.write('Lower AURC is better. Adding technical quality worsened Hamming AURC in both cohorts, both resolutions and both modes. All paired 320−224 Hamming AURC intervals include zero; 200 whole-component bootstrap replicates are conditional and not selection-adjusted.')
    with figures:
        st.caption('Original reviewed aggregate figures; no patient records. Calibration panels are in sample; validation panels reuse development data. Sparse bins are unstable.')
        for name, record in data['figures'].items():
            st.image(str(ROOT / 'results/calibration-v15' / name), caption=record['cohort_interpretation'])
    with critical:
        st.subheader('Findings to defend, not hide')
        st.write('NLL/Brier fell slightly while ECE increased in all four cohort/resolution combinations. Neither scalar calibration nor a favorable macro metric establishes better disease detection or clinical safety.')
        st.write('320 px has class-specific harms: inspect C false positives and A/G recall rather than announcing a universal winner. H has only 16 validation positives; mixed O and inherited eye labels limit interpretation.')
        st.write('Possible causes—resolution loss, image quality and label ambiguity—are hypotheses. This evidence does not isolate their causal contributions. A new experiment must specify one factor, matched controls, selection rules and uncertainty before looking at outcomes.')
        st.write('Head-only MC omits backbone uncertainty. Remapped variance can decrease without demonstrating less epistemic uncertainty. Quality ranks are technical proxies, not clinical grades. Grad-CAM does not prove lesion localization.')
    with viva:
        for question, answer in VIVA:
            with st.expander(question):
                st.write(answer)
        st.write('Reproduction has three levels: public evidence/hash checks; synthetic CPU software tests; separately authorized repetition of research with restricted inputs. Passing one level does not establish the next.')
        st.write('Native Mac execution, durable private backup and independent all-eight confirmation remain separate gaps. Cloud loopback is temporary workspace access, not permanent hosting or a laptop-accessible deployment.')
        st.code(json.dumps({'aggregate_sha256': AGGREGATE_SHA256,
                            'protocol_sha256': data['provenance']['protocol_sha256'],
                            'analysis_source_commit': data['provenance']['analysis_source_commit']}, indent=2), language='json')
        guide = ROOT / 'docs/ACADEMIC_RESEARCH_DEFENCE.md'
        if guide.is_file():
            st.download_button('Download research defence guide', guide.read_text(),
                               file_name='trustretina-research-defence.md', mime='text/markdown')
