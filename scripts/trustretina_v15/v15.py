"""CPU post-freeze analysis of saved predictions; no model or runner imports."""
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from trustretina.config import CLASSES
from trustretina.evaluation.threshold_cv import summarize, support
from trustretina.reliability.development import (
    calibration_diagnostics, coverage_curve, fit_shared_temperature,
    fixed_coverage_summary, patient_quality, quality_midrank, signals,
    temperature_probabilities,
)
from .v15_inputs import FIXTURE, RUNS, load_bundle, sha256

LIMITATIONS = [
    'Synthetic checks are software evidence only; never real V15 results.',
    'Calibration fit diagnostics are in sample; validation was reused for model selection.',
    'Calibration H has only 10 positive patients; validation H has 16. Shared T gives no class-conditional guarantee.',
    'Patient-derived ODIR labels are not verified eye diagnoses; H/O and external-label semantics remain limited.',
    'Containment components can contain mixed patient labels; no voting or relabeling.',
    'Head-only MC variability omits backbone uncertainty; quality proxies are not clinical gradability.',
    'Confidence/quality fusion can worsen reliability; retain every fixed ablation and regression.',
    'No independent all-eight confirmation or clinical acceptance policy is established.',
]
STRATEGIES = ('confidence', 'confidence_uncertainty', 'confidence_uncertainty_quality')


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(serializable(value), indent=2, sort_keys=True, allow_nan=False)+'\n')


def diagnostic(y, p, groups):
    result = calibration_diagnostics(y, p, groups)
    result['class_order'] = CLASSES
    for j, row in enumerate(result['per_class']):
        row['label'] = CLASSES[j]
        row['nll'] = float(-(y[:, j]*np.log(p[:, j].clip(1e-12, 1-1e-12))
                             +(1-y[:, j])*np.log1p(-p[:, j].clip(1e-12, 1-1e-12))).mean())
        row['sparse_or_single_outcome'] = min(row['positive'], row['negative']) < 20
    return result


def descriptive(y, p, samples, quality, groups):
    scores = signals(p, samples, quality)
    counts = summarize(y, p >= .5)
    curves = {}
    for name in STRATEGIES:
        curves[name] = {**coverage_curve(y, p, scores[name]),
                        'fixed_coverage': fixed_coverage_summary(y, p, scores[name]),
                        'interpretation': 'Retrospective descriptive subsets; every actual decision remains REVIEW REQUIRED'}
    variance = samples.var(0, ddof=0)
    mean = samples.mean(0)
    clipped = mean.clip(1e-12, 1-1e-12)
    entropy = -(clipped*np.log(clipped)+(1-clipped)*np.log1p(-clipped))
    return {'calibration': diagnostic(y, p, groups), 'classification': counts,
            'supplying_support': support(y, groups), 'curves': curves,
            'signals': scores, 'mc_mean': mean, 'mc_population_variance': variance,
            'mc_predictive_binary_entropy': entropy,
            'quality_mean': float(quality.mean()),
            'quality_feature_definition': ['foreground_fraction', 'contrast', 'gradient_energy']}


def paired_differences(candidate, reference):
    result = {}
    for label, c, r in zip(CLASSES, candidate['classification']['per_class'], reference['classification']['per_class']):
        result[label] = {k: c[k]-r[k] if c[k] is not None and r[k] is not None else None
                         for k in ('tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1',
                                   'false_positive_rate', 'false_negative_rate')}
        result[label]['point_regression'] = bool(c['fp'] > r['fp'] or c['fn'] > r['fn'] or c['f1'] < r['f1'])
    return {'direction': '320 minus 224; descriptive paired patients', 'per_class': result,
            'calibration': {k: candidate['calibration'][k]-reference['calibration'][k]
                            for k in ('patient_nll', 'component_equal_nll', 'brier',
                                      'component_equal_brier', 'macro_ece_10_equal_width_bins')}}


def bootstrap_curves(arms, groups):
    """200 paired whole-component resamples, seed 2026, no policy selection."""
    groups = np.asarray(groups); unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError('At least two components required for descriptive uncertainty')
    mapping = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(2026)
    values = {(arm, strategy): [] for arm in RUNS for strategy in STRATEGIES}
    for _ in range(200):
        idx = np.concatenate([mapping[i] for i in rng.integers(0, len(unique), len(unique))])
        for arm, (y, p, scores) in arms.items():
            for strategy in STRATEGIES:
                values[arm, strategy].append(coverage_curve(y[idx], p[idx], scores[strategy][idx])['hamming_aurc'])
    return {strategy: {
        '320_hamming_aurc_percentile_95': np.quantile(values[RUNS[0], strategy], [.025, .975]),
        '224_hamming_aurc_percentile_95': np.quantile(values[RUNS[1], strategy], [.025, .975]),
        '320_minus_224_hamming_aurc_percentile_95': np.quantile(
            np.asarray(values[RUNS[0], strategy])-values[RUNS[1], strategy], [.025, .975]),
        'replicates': 200, 'seed': 2026, 'unit': 'whole containment component',
        'interpretation': 'Conditional descriptive interval; not selection-adjusted or a clinical guarantee'}
        for strategy in STRATEGIES}


def analyze(manifest_path, manifest_sha256, protocol_sha256, approval, *, fixture=False):
    """No fitting until BOTH arms, ALL roles, and ALL artifacts pass validation."""
    bundle = load_bundle(manifest_path, manifest_sha256, protocol_sha256, approval, fixture=fixture)
    # Preflight both fitting sets before fitting either scalar.
    for item in bundle.values():
        d = item['data']['calibration']; y = d['labels']
        if len(set(d['component_ids'])) < 50 or np.any(y.sum(0) == 0) or np.any(y.sum(0) == len(y)):
            raise ValueError('Shared calibration requires >=50 components and both outcomes in every class')
    report = {'schema_version': 15, 'data_kind': FIXTURE if fixture else 'real_v14',
              'manifest_sha256': manifest_sha256, 'protocol_sha256': protocol_sha256,
              'class_order': CLASSES, 'decision_threshold': .5,
              'deployment_default': 'REVIEW REQUIRED', 'risk_policy_fitted': False,
              'threshold_fitted': False, 'historical_final_accessed': False,
              'training': False, 'model_inference': False, 'gpu_requested': False,
              'limitations': LIMITATIONS, 'runs': {}, 'paired_comparisons': {}}
    for run_id, item in bundle.items():
        d = item['data']['calibration']
        fit = fit_shared_temperature(d['probabilities'], d['labels'], d['component_ids'],
                                     role='calibration', allow_fitting=True)
        result = {'fit': fit, 'checkpoint_sha256': item['fit']['artifact_sha256']['best.pt'],
                  'kernel_version': item['fit']['kernel_version'], 'roles': {}}
        for role, d in item['data'].items():
            q = patient_quality(item['roles'][role], quality_midrank(item['quality']['train'], item['quality'][role]), d['patient_ids'])
            raw = d['probabilities']; calibrated = temperature_probabilities(raw, fit['temperature'])
            if not np.array_equal(raw >= .5, calibrated >= .5):
                raise ValueError('Numerical threshold crossing; analysis blocked')
            modes = {'raw': (raw, d['passes']), 'calibrated': (calibrated, np.stack([
                temperature_probabilities(p, fit['temperature']) for p in d['passes']]))}
            result['roles'][role] = {'interpretation': 'in-sample fit diagnostic' if role == 'calibration'
                                    else 'reused-validation development diagnostic, not independent confirmation'}
            for mode, (p, samples) in modes.items():
                result['roles'][role][mode] = descriptive(d['labels'], p, samples, q, d['component_ids'])
            result['roles'][role]['all_binary_decisions_unchanged'] = True
        # Persist each resolution's TRAIN-only ECDF; never transfer the old demo normalizer.
        result['sorted_training_quality_distributions'] = np.sort(item['quality']['train'], axis=0)
        report['runs'][run_id] = result
    for role in ('validation', 'calibration'):
        report['paired_comparisons'][role] = {}
        for mode in ('raw', 'calibrated'):
            a, b = [report['runs'][r]['roles'][role][mode] for r in RUNS]
            paired = paired_differences(a, b)
            arms = {}
            for r in RUNS:
                d = bundle[r]['data'][role]; t = report['runs'][r]['fit']['temperature']
                p = d['probabilities'] if mode == 'raw' else temperature_probabilities(d['probabilities'], t)
                arms[r] = (d['labels'], p, report['runs'][r]['roles'][role][mode]['signals'])
            paired['descriptive_aurc_intervals'] = bootstrap_curves(arms, bundle[RUNS[0]]['data'][role]['component_ids'])
            report['paired_comparisons'][role][mode] = paired
    report['packages'] = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn')}
    report['packages']['python'] = platform.python_version()
    import trustretina.reliability.development as shared
    import trustretina.evaluation.threshold_cv as counts
    report['software_sha256'] = {p.name: sha256(p) for p in (
        Path(__file__), Path(__file__).with_name('v15_inputs.py'),
        Path(shared.__file__), Path(counts.__file__))}
    return report


def save_report(report, output):
    """Fresh private output only. No frozen bundle is replaced."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'analysis.json', report)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    for run_id, run in report['runs'].items():
        for role, result in run['roles'].items():
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for j, ax in enumerate(axes.flat):
                for mode in ('raw', 'calibrated'):
                    bins = result[mode]['calibration']['per_class'][j]['bins']
                    points = [b for b in bins if b['n']]
                    ax.plot([b['mean_probability'] for b in points], [b['positive_fraction'] for b in points], 'o-', label=mode)
                ax.plot([0, 1], [0, 1], 'k--'); ax.set(title=CLASSES[j], xlim=(0, 1), ylim=(0, 1))
            axes.flat[0].legend()
            fig.suptitle(f"{report['data_kind']} | {run_id} | {role}: {result['interpretation']}", fontsize=8)
            fig.tight_layout(); fig.savefig(output/f'{run_id}-{role}-reliability.png'); plt.close(fig)
            fig, ax = plt.subplots()
            for mode in ('raw', 'calibrated'):
                for name, curve in result[mode]['curves'].items():
                    ax.step(curve['coverage'], curve['hamming_risk'], where='pre', label=mode+' '+name)
            ax.set(xlabel='Descriptive patient coverage', ylabel='Hamming risk', ylim=(0, 1),
                   title=report['data_kind']+'; no acceptance policy')
            ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(output/f'{run_id}-{role}-risk-coverage.png'); plt.close(fig)
    write_json(output/'output-hashes.json', {p.name: sha256(p) for p in sorted(output.iterdir())})
