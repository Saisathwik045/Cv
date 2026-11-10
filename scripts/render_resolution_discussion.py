#!/usr/bin/env python3
"""Readable prespecified tables from complete verified aggregates; no new analysis."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/resolution-v13';p=json.loads((out/'comparison.json').read_text());e=json.loads((out/'execution-audit.json').read_text())
    if not e['complete_matrix'] or len(p['folds'])!=3 or len(p['sources'])!=6:raise ValueError('No paired conclusion from incomplete matrix')
    arms=('res224','res320');lines=['# V13 matched resolution findings','',
      'All six fresh-ImageNet ConvNeXt-Tiny ratio-BCE fits completed. The unchanged training-only containment folds, common microbatch8/accumulation2, inner-only checkpoint selection, raw mean-eye sigmoid predictor and fixed p≥.5 define this comparison. V11 checkpoints were not used as references. No historical final, calibration, threshold fitting, full-data refit or demo change.','',
      '| Arm | Mean outer macro AUROC | Mean macro AP | Mean macro F1 | Mean micro F1 |','|---|---:|---:|---:|---:|']
    for arm in arms:
        m=p['mean_fold_metrics'][arm];lines.append(f"| {arm} | {m['macro_auroc']:.5f} | {m['macro_auprc_average_precision']:.5f} | {m['macro_f1']:.5f} | {m['micro_f1']:.5f} |")
    lines+=['','| Paired mean difference (320−224) | Estimate | Conditional 95% interval | Defined / total |','|---|---:|---:|---:|']
    for k in ('macro_auroc','macro_auprc_average_precision','macro_f1','H_recall','H_f1'):
        d=sum(f['paired_differences'][k] for f in p['folds'])/3 if k.startswith('H_') else p['mean_fold_metrics']['res320'][k]-p['mean_fold_metrics']['res224'][k];b=p['paired_component_bootstrap'][k];ci=b['percentile_95_interval'];text='undefined' if ci is None else f'[{ci[0]:+.5f}, {ci[1]:+.5f}]'
        lines.append(f"| {k} | {d:+.5f} | {text} | {b['defined_replicates']} / {b['total_replicates']} |")
    ci=p['paired_component_bootstrap']['macro_auroc']['percentile_95_interval']
    if ci is None or ci[0]<=0<=ci[1]:conclusion='The primary interval does not establish a resolution improvement.'
    elif ci[0]>0:conclusion='The conditional paired primary interval favors320 on this development cohort; this is not independent confirmation.'
    else:conclusion='The conditional paired primary interval favors224 on this development cohort; increasing resolution did not improve the primary endpoint.'
    df1=p['mean_fold_metrics']['res320']['macro_f1']-p['mean_fold_metrics']['res224']['macro_f1']
    lines+=['',conclusion+(' Mean fixed-threshold macro F1 is lower for320; do not make a blanket improvement claim.' if df1<0 else ' Any macro F1 increase must be considered alongside all-class false-positive/false-negative tradeoffs.'),
      f"The prespecified numerical rule nominates {p['candidate_for_future_study']} for future study only. There is no global fold-checkpoint selection, deployment decision or automatic refit.",'',
      'Intervals use2000 paired component-cluster bootstrap replicates, fixed seed2026, sampling within each outer fold with the same groups for both arms. They condition on fitted models and omit training/checkpoint-selection variability. Previous adaptive choices, same-source data, three folds and one seed limit independence. Outer sample separation inside each fit does not erase prior development exposure.','',
      '## Every class: descriptive pooled outer decisions','',
      'All2275 training patients contribute once per arm. Each arm pools predictions from three separately trained models; pooled ranking metrics are secondary. See per-class.csv for every fold/arm row and comparison.json for support/undefined flags.','',
      '| Class | Arm | Positive / negative | AUROC | AP | Precision | Recall | F1 | TP | FP | FN |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label in 'NDGCAHMO':
        for arm in arms:
            m=p['secondary_pooled'][arm];r=next(r for r in m['confusion'] if r['label']==label);x=m['per_class'][label]
            f=lambda v:'undefined' if v is None else f'{v:.4f}'
            lines.append(f"| {label} | {arm} | {r['positive']} / {r['negative']} | {f(x['auroc'])} | {f(x['average_precision'])} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['tp']} | {r['fp']} | {r['fn']} |")
    lines+=['','## H and O tradeoffs','']
    for label in ('H','O'):
        rows=[next(r for r in p['secondary_pooled'][a]['confusion'] if r['label']==label) for a in arms]
        a,b=rows
        lines.append(f"- {label}: recall {a['recall']:.4f}→{b['recall']:.4f}, F1 {a['f1']:.4f}→{b['f1']:.4f}; TP {a['tp']}→{b['tp']}, FP {a['fp']}→{b['fp']}, FN {a['fn']}→{b['fn']}, among {a['positive']} positive patients. These are fixed-threshold development counts, not a clinical sensitivity guarantee.")
    lines+=['','H has only23/22/22 outer positives and66 positive containment components overall. Sparse support increases uncertainty but is not a sole explanation for its errors. Patient-derived labels are inherited by both eye targets; H/O case-definition and localization limitations from v12 persist. Resolution changes neither source labels nor clinical validity. Negative findings and class-specific harms must be retained.','',
      '## Execution, accounting and recovery','',
      'See execution-audit.json for actual versions, source/nonce/output hashes, dependencies and worker times; execution-provenance.json records settled quota counters separately. Best/last checkpoints, aligned eye/patient predictions, histories and logs remain in pinned private Kaggle output versions and private local artifacts, outside Git. The approved scientific worker/configuration/protocol/reporter hashes are unchanged.','',
      'Next research decisions need separate authorization and a fresh suitable independent paired-eye/all-eight labeled evaluation cohort with compatible definitions and overlap exclusion. Do not tune on the historical final holdout, transfer its calibration to these checkpoints, deploy exploratory thresholds or replace the working demo. REVIEW REQUIRED remains unchanged.','',
      '[Frozen protocol](../../docs/RESOLUTION_V13.md) · [Complete paired aggregates](comparison.json) · [Per-fold per-class CSV](per-class.csv) · [Execution audit](EXECUTION.md)','',
      '![Paired metrics](paired-resolution-metrics.png)','','![H/O tradeoffs](H-O-tradeoffs.png)','','![All-eight FP/FN](all-eight-error-counts.png)','']
    (out/'DISCUSSION.md').write_text('\n'.join(lines))
    (out/'RESULTS.md').write_text('\n'.join(lines[:lines.index('## Every class: descriptive pooled outer decisions')])+ '\n[All-eight per-class metrics, H/O tradeoffs, plots and limitations](DISCUSSION.md) · [Verified execution](EXECUTION.md) · [Settled accounting and artifact hashes](execution-provenance.json)\n');print('Complete-matrix discussion generated from verified aggregates only.')


if __name__=='__main__':main()
