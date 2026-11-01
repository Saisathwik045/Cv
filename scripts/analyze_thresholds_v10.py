#!/usr/bin/env python3
"""One CPU-only analysis of pinned validation predictions. No inference or jobs."""

import argparse
import csv
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from trustretina.data.validation import file_hash
from trustretina.evaluation.threshold_cv import make_group_folds, evaluate_folds
from report_controlled_results import read_predictions

PREREGISTERED = '1d1516f1f1687b2b49a4962287b04b8a1b36ce2e'
CONFIG = ROOT / 'configs/threshold-v10.toml'
PROTOCOL = ROOT / 'docs/THRESHOLD_V10.md'
PRIVATE = ROOT / 'artifacts/analysis/threshold-v10'
PUBLIC = ROOT / 'results/threshold-v10'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def preregistration():
    subprocess.run(['git', 'merge-base', '--is-ancestor', PREREGISTERED, 'HEAD'], cwd=ROOT, check=True)
    for path in (CONFIG, PROTOCOL):
        saved = subprocess.check_output(['git', 'show', PREREGISTERED + ':' + str(path.relative_to(ROOT))], cwd=ROOT)
        if saved != path.read_bytes():
            raise ValueError('Prospective protocol/configuration changed; no analysis permitted')
    return tomllib.loads(CONFIG.read_text())


def demo_hashes():
    # Deliberately do not open historical final/calibration datasets or reports.
    expected = json.loads((ROOT / 'results/round2-v9/execution-provenance.json').read_text())['protected_sha256']
    paths = [p for p in expected if p.startswith('artifacts/inference-v8/')
             or p == 'results/reliability-v6/candidate-contract.json']
    observed = {p: file_hash(ROOT / p) for p in paths}
    if any(observed[p] != expected[p] for p in paths):
        raise ValueError('Frozen demo bundle changed before/after analysis')
    return observed


def csv_rows(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def publish(report):
    PUBLIC.mkdir(parents=True, exist_ok=True)
    write(PUBLIC / 'aggregate.json', report)
    rows = []
    for j, (base, cv) in enumerate(zip(report['baseline']['per_class'], report['threshold_cv']['per_class'])):
        stability = report['stability'][j]
        for strategy, item in [('fixed_0.5', base), ('heldout_threshold_cv', cv)]:
            rows.append({'strategy': strategy, **item,
                         'delta_tp_vs_fixed': item['tp'] - base['tp'],
                         'delta_fp_vs_fixed': item['fp'] - base['fp'],
                         'delta_fn_vs_fixed': item['fn'] - base['fn'],
                         'delta_f1_vs_fixed': item['f1'] - base['f1'],
                         'fold_thresholds': '|'.join(str(t) for t in stability['fold_thresholds']),
                         **{k: stability[k] for k in ('minimum', 'q1', 'median', 'q3', 'maximum', 'iqr', 'distinct_values', 'fallback_folds')}})
    csv_rows(PUBLIC / 'per-class.csv', rows)
    lines = [
        '# Exploratory group-aware threshold analysis v10', '',
        'CPU-only; generated from verified saved development predictions. The selected ConvNeXt cosine epoch-13 model is fixed. '
        'All 525 patients / 518 containment components are reused; no historical final or calibration datasets were opened. '
        'No old temperature was applied, model inference performed, or demo replaced.', '',
        f"[Prospective protocol](../../docs/THRESHOLD_V10.md) and configuration were committed at `{PREREGISTERED}` before computing this analysis. "
        'Four fixed seed-2026 multilabel containment-group folds use the existing rarest-label allocator. '
        'Each threshold is fitted on three folds and assessed on its untouched threshold-assessment fold. '
        'Objective: per-class F1 on .05:.05:.95; exact ties nearest .5, then higher; fewer than 10 supplying fitting components for either outcome retains .5. '
        'Predictions use p ≥ threshold and patient weights.', '',
        '**These are not independent model-assessment folds.** All validation labels already influenced checkpoint/backbone selection. '
        'Cross-validation protects the threshold-fitting step only; it cannot remove prior model-selection bias. '
        'This remains exploratory and provides no independent confirmation.', '',
        '| Procedure | Macro F1 | Micro F1 | TP | FP | FN | Exact match | Labelwise accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name, key in [('Fixed 0.5', 'baseline'), ('Pooled held-out threshold CV', 'threshold_cv')]:
        m = report[key]
        lines.append(f"| {name} | {m['macro_f1']:.4f} | {m['micro_f1']:.4f} | {m['total_tp']} | {m['total_fp']} | {m['total_fn']} | {m['subset_exact_match_accuracy']:.4f} | {m['labelwise_binary_accuracy']:.4f} |")
    lines += ['', 'Exact match requires all eight labels correct; labelwise accuracy averages correct bits and is not diagnostic accuracy. '
              'Macro F1 averages eight F1s; micro F1 pools TP/FP/FN on (n,8) arrays. '
              'Zero-denominator metrics use zero with defined flags retained in the aggregate/CSV.', '',
              '## Per-class held-out outcomes', '',
              '| Class | + / − | Procedure | Precision | Recall | F1 | TP | FP | FN |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['label']} | {row['positive']} / {row['negative']} | {row['strategy']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['tp']} | {row['fp']} | {row['fn']} |")
    lines += ['', '## Threshold stability', '',
              '| Class | Folds 0 / 1 / 2 / 3 | Min–max | Median | Q1–Q3 | IQR | Fallback fits |',
              '|---|---|---|---:|---|---:|---:|']
    for row in report['stability']:
        values = ' / '.join(f'{t:.2f}' for t in row['fold_thresholds'])
        lines.append(f"| {row['label']} | {values} | {row['minimum']:.2f}–{row['maximum']:.2f} | {row['median']:.3f} | {row['q1']:.3f}–{row['q3']:.3f} | {row['iqr']:.3f} | {row['fallback_folds']} |")
    lines += ['', 'The four fits overlap; their ranges/IQR describe sensitivity to fitting rows, not confidence intervals or four independent replications. '
              'These fold-specific thresholds are not a deployable all-data vector. No all-development refit was performed.', '',
              '## H: sparse-support examination', '',
              '| Fold | Fit + / − groups | Threshold | Fallback | Held-out + / − | Baseline TP / FP / FN | CV TP / FP / FN | CV precision / recall / F1 |',
              '|---|---|---:|---|---|---|---|---|']
    for fold in report['folds']:
        fit = fold['fitting'][5]
        base = fold['baseline']['per_class'][5]
        cv = fold['threshold_cv']['per_class'][5]
        lines.append(f"| {fold['fold']} | {fit['positive_components']} / {fit['negative_components']} | {fit['threshold']:.2f} | {fit['fallback']} | {cv['positive']} / {cv['negative']} | {base['tp']} / {base['fp']} / {base['fn']} | {cv['tp']} / {cv['fp']} / {cv['fn']} | {cv['precision']:.3f} / {cv['recall']:.3f} / {cv['f1']:.3f} |")
    lines += ['', 'Only 16 H-positive patients exist overall. One changed prediction can substantially move per-fold recall. '
              'No clinical recall target, safety claim or acceptance threshold follows from F1 optimization.', '',
              '## Observed tradeoffs', '']
    for row in rows:
        if row['strategy'] != 'heldout_threshold_cv':
            continue
        lines.append(f"- {row['label']}: ΔFP {row['delta_fp_vs_fixed']:+d}, ΔFN {row['delta_fn_vs_fixed']:+d}, ΔTP {row['delta_tp_vs_fixed']:+d}, ΔF1 {row['delta_f1_vs_fixed']:+.4f} versus .5. Positive ΔFN means more missed positives; negative ΔFN means fewer. Positive ΔFP means more false alarms.")
    lines += ['', '![Paired errors and fold thresholds](threshold-tradeoffs.png)', '',
              '## Recommendation and limitations', '',
              'Retain the original frozen research demo and REVIEW REQUIRED behavior. Do not install these exploratory fold thresholds. '
              'Use the class-specific count/stability findings to document costs and sparse support, not to invent a clinical recall target. '
              'Before deploying a new checkpoint, prospectively define the threshold objective with domain stakeholders and obtain model-development predictions '
              'independent of checkpoint selection (for example training-only group-aware out-of-fold model predictions in a separately authorized study), '
              'then freeze the complete new pipeline for evaluation on a fresh suitable independent cohort. '
              'Probability calibration, if later authorized for the new checkpoint, needs a separate prospective fit; the old temperature is incompatible. '
              'No further experiment or data collection is launched here.', '',
              'Patient-derived labels are not independently verified eye diagnoses. Label-stratified fold assignment uses labels but never model scores. '
              'AUROC/AP are unchanged because probability rankings are unchanged. Fold support and raw fold-level outcomes, including negative results, '
              'are in [aggregate.json](aggregate.json); [per-class CSV](per-class.csv) contains all counts and defined flags. '
              'Private sample-aligned assignments/OOF decisions and their hashes are recorded in the provenance. No final-holdout access, calibration fit, '
              'automatic acceptance policy, model training, GPU job or paid resource was used.', '']
    (PUBLIC / 'RESULTS.md').write_text('\n'.join(lines))
    plot(report)


def plot(report):
    os.environ.setdefault('MPLCONFIGDIR', str(PRIVATE / 'mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(8)
    for ax, count, title in zip(axes[:2], ('fp', 'fn'), ('False positives', 'Missed positives (FN)')):
        ax.bar(x - .18, [r[count] for r in report['baseline']['per_class']], .36, label='Fixed 0.5')
        ax.bar(x + .18, [r[count] for r in report['threshold_cv']['per_class']], .36, label='Held-out threshold CV')
        ax.set_title(title)
        ax.set_xticks(x, [r['label'] for r in report['stability']])
        ax.legend(fontsize=8)
    for k in range(4):
        axes[2].scatter(x + (k - 1.5) * .07, [r['fold_thresholds'][k] for r in report['stability']], label=f'Fold {k}')
    axes[2].axhline(.5, color='grey', linestyle='--', label='Baseline')
    axes[2].set_xticks(x, [r['label'] for r in report['stability']])
    axes[2].set_ylim(0, 1)
    axes[2].set_title('Four overlapping fitting-fold thresholds')
    axes[2].legend(fontsize=8)
    fig.suptitle('Exploratory reused validation: no independent confirmation')
    fig.tight_layout()
    fig.savefig(PUBLIC / 'threshold-tradeoffs.png', dpi=150)
    plt.close(fig)


def main(attempt):
    cfg = preregistration()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    destination = PRIVATE / f'attempt-{attempt}'
    for completed in PRIVATE.glob('attempt-*/complete.json'):
        marker = json.loads(completed.read_text())
        for name, digest in marker['private_files'].items():
            if file_hash(completed.parent / name) != digest:
                raise ValueError('Completed private result changed')
        for name, digest in marker['public_files'].items():
            if file_hash(PUBLIC / name) != digest:
                raise ValueError('Completed public result changed')
        print('Completed fixed analysis exists and hashes verify; no recomputation or fitting performed.')
        return
    destination.mkdir(exist_ok=False)
    start = time.perf_counter()
    try:
        before = demo_hashes()
        recorded = json.loads((ROOT / 'results/round2-v9/verified-comparison.json').read_text())
        expected = recorded['sources'][cfg['run_id']]
        registry = json.loads((ROOT / 'results/round2-v9/queue.json').read_text())['registry']
        run = next(r for r in registry['runs'] if r['id'] == cfg['run_id'])
        report, folder, verification, data, metrics, _, _ = read_predictions(ROOT / run['attempts'][-1]['bundle'])
        if (report['run_id'] != cfg['run_id'] or report['backbone'] != 'convnext_tiny'
                or report['selection']['best_epoch'] + 1 != 13
                or report['output_hashes']['best.pt'] != cfg['checkpoint_sha256']
                or verification['kernel_version'] != cfg['kernel_version']
                or verification['artifact_sha256'] != expected['report_sha256']
                or file_hash(folder / 'validation-patients.npz') != expected['patient_predictions_sha256']
                or len(data['labels']) != 525 or len(set(data['component_ids'])) != 518
                or metrics['class_order'] != cfg['class_order']):
            raise ValueError('Pinned selected checkpoint/predictions/cohort identity mismatch')
        folds, repairs = make_group_folds(data['labels'], data['patient_ids'], data['component_ids'])
        analysis, decisions, thresholds = evaluate_folds(
            data['labels'], data['probabilities'], data['patient_ids'], data['component_ids'], folds,
            role=cfg['role'], allow_exploratory=True)
        for key, old_key in [('macro_f1', 'macro_f1'), ('micro_f1', 'micro_f1')]:
            if not np.isclose(analysis['baseline'][key], metrics[old_key], atol=1e-12, rtol=0):
                raise ValueError('Fixed-.5 baseline failed reproduction')
        if demo_hashes() != before:
            raise ValueError('Frozen demo changed during analysis')
        np.savez_compressed(destination / 'oof-decisions.npz', **data, fold_ids=folds,
                            baseline_decisions=data['probabilities'] >= .5,
                            heldout_threshold_decisions=decisions, fold_thresholds=thresholds)
        write(destination / 'fold-assignment.json', {
            'patients': [{'patient_id': str(p), 'component_id': str(g), 'fold': int(k)}
                         for p, g, k in zip(data['patient_ids'], data['component_ids'], folds)],
            'support_repairs': repairs})
        full_analysis = analysis
        write(destination / 'full-analysis.json', full_analysis)
        # Candidate fit curves are private; aggregate fold supports/thresholds remain public.
        analysis = json.loads(json.dumps(full_analysis))
        for fold in analysis['folds']:
            for fitting in fold['fitting']:
                fitting.pop('candidates')
        provenance = {'preregistered_commit': PREREGISTERED,
                      'protocol_sha256': file_hash(PROTOCOL), 'config_sha256': file_hash(CONFIG),
                      'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      'implementation_sha256': {p: file_hash(ROOT / p) for p in (
                          'scripts/analyze_thresholds_v10.py', 'src/trustretina/evaluation/threshold_cv.py',
                          'src/trustretina/data/splits.py', 'scripts/report_controlled_results.py',
                          'src/trustretina/runner/verified_artifacts.py')},
                      'source': expected, 'kernel_version': verification['kernel_version'],
                      'checkpoint_sha256': cfg['checkpoint_sha256'],
                      'input_probabilities': cfg['probabilities'], 'python': platform.python_version(),
                      'dependencies': {name: importlib.metadata.version(name) for name in ('numpy', 'scikit-learn', 'matplotlib')},
                      'cpu_analysis_wall_seconds_excluding_publication': time.perf_counter() - start,
                      'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'demo_sha256_unchanged': before,
                      'private_files': {p.name: file_hash(p) for p in destination.iterdir() if p.is_file()},
                      'historical_final_opened': False, 'calibration_data_opened': False,
                      'calibration_temperature_applied': False, 'gpu_jobs': 0, 'model_inference': False,
                      'all_data_threshold_refit': False, 'demo_replaced': False,
                      'patient_group_separation_passed': all(f['group_overlap'] == 0 for f in analysis['folds']),
                      'support_repairs': len(repairs)}
        result = {'analysis': cfg['analysis'], 'config': cfg, 'provenance': provenance, **analysis}
        publish(result)
        write(destination / 'provenance.json', provenance)
        write(destination / 'complete.json', {
            'private_files': {p.name: file_hash(p) for p in destination.iterdir() if p.is_file()},
            'public_files': {p.name: file_hash(p) for p in PUBLIC.iterdir() if p.is_file()}})
        print(json.dumps({k: analysis[k] for k in ('baseline', 'threshold_cv', 'stability')}, indent=2))
    except Exception as error:
        write(destination / 'failed.json', {'error_type': type(error).__name__, 'message': str(error),
                                            'elapsed_seconds': time.perf_counter() - start})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, default=1, help='New attempt only after a preserved implementation failure')
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error('Positive attempt number required')
    main(args.attempt)
