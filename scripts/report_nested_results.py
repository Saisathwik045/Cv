#!/usr/bin/env python3
"""Verified six-fit outer comparison; checkpointed CPU bootstrap, no fitting/jobs."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import nested_experiment as c
from trustretina.runner.verified_artifacts import read_stage
from trustretina.data.validation import canonical_hash,file_hash
from trustretina.training.nested import patient_rows
from trustretina.training.comparison import comparison_metrics
from trustretina.evaluation.metrics import aggregate_patient_mean
from trustretina.evaluation.nested_comparison import paired_summary,bootstrap_replicate,bootstrap_intervals


def main(batch):
    p=json.loads((c.BASE/'registry.json').read_text());manifest=c.verify_plan(p)
    if any(r['status']!='complete' for r in p['runs']):raise ValueError('Six completed fits required; no partial comparison/selection')
    train_rows=json.loads((ROOT/p['cache']['training_metadata_file']).read_text());pairs=[{} for _ in range(3)];sources={};histories={}
    for run in p['runs']:
        report,folder,v=read_stage(ROOT/run['attempts'][-1]['bundle'])
        if report['status']!='complete' or not report.get('strict_best_reload') or not report.get('outer_assessed') or report['plan_sha256']!=p['plan_sha256'] or report['run_id']!=run['id'] or report['arm']!=run['arm'] or report['outer_fold']!=run['outer_fold'] or report['provenance']['fold_manifest_sha256']!=p['fold_manifest_sha256']:raise ValueError('Wrong/unfrozen outer report')
        with np.load(folder/'outer-patients.npz',allow_pickle=False) as f:d={k:f[k] for k in ('labels','probabilities','patient_ids','component_ids')}
        if comparison_metrics(d['labels'],d['probabilities'])!=report['metrics']['patient']:raise ValueError('Outer metric reproduction failed')
        with np.load(folder/'outer-images.npz',allow_pickle=False) as f:
            aggregate=aggregate_patient_mean(f['patient_ids'].tolist(),f['eyes'].tolist(),f['labels'],f['probabilities'],f['component_ids'].tolist())
        if any(not np.array_equal(d[k],aggregate[k]) for k in d):raise ValueError('Mean-eye predictor mismatch')
        expected=patient_rows([train_rows[i] for i in manifest['folds'][run['outer_fold']]['outer_indices']])
        for key,field in [('patient_ids','patient_id'),('component_ids','component_id'),('labels','labels')]:
            if not np.array_equal(d[key],np.asarray([r[field] for r in expected])):raise ValueError('Outer rows differ from frozen training-only assignment')
        pairs[run['outer_fold']][run['arm']]=d
        sources[run['id']]={'kernel_version':v['kernel_version'],'report_sha256':v['artifact_sha256'],'best_sha256':report['output_hashes']['best.pt'],'last_sha256':report['output_hashes']['last.pt'],'outer_predictions_sha256':file_hash(folder/'outer-patients.npz'),'best_epoch':report['selection']['best_epoch']+1}
        histories[run['id']]=report['history']
    summary=paired_summary(pairs)
    if summary['patients']!=2275 or summary['components']!=2240:raise ValueError('Outer coverage differs from original TRAIN cohort')
    analysis=c.BASE/'analysis';analysis.mkdir(exist_ok=True);identity=canonical_hash({'sources':sources,'plan':p['plan_sha256']})
    checkpoint=analysis/'bootstrap.json'
    saved=json.loads(checkpoint.read_text()) if checkpoint.exists() else {'identity':identity,'replicates':[]}
    if saved['identity']!=identity:raise ValueError('Bootstrap input changed; preserve previous analysis')
    target=p['config']['bootstrap_replicates'];records=saved['replicates']
    for i in range(len(records),min(target,len(records)+batch)):
        records.append(bootstrap_replicate(pairs,i,p['config']['bootstrap_seed']))
        if len(records)%20==0:c.write(checkpoint,saved)
    c.write(checkpoint,saved)
    if len(records)<target:
        print(json.dumps({'status':'CPU bootstrap paused; rerun same command to continue','completed_replicates':len(records),'required':target}));return
    summary.update(plan_sha256=p['plan_sha256'],sources=sources,paired_component_bootstrap=bootstrap_intervals(records),
                   bootstrap_checkpoint_sha256=file_hash(checkpoint),historical_final_accessed=False,calibration_fitted=False,threshold=.5,independent_confirmation=False)
    out=ROOT/'results/nested-v11';out.mkdir(exist_ok=True);c.write(out/'comparison.json',summary)
    rows=[]
    for f in summary['folds']:
        for arm,m in f['arms'].items():
            for j,(label,metrics) in enumerate(m['per_class'].items()):
                confusion=next(x for x in m['confusion'] if x['label']==label);support=next(x for x in m['component_support'] if x['label']==label)
                rows.append({'outer_fold':f['outer_fold'],'arm':arm,'label':label,**metrics,**confusion,'positive_components':support['positive_components'],'negative_components':support['negative_components']})
    with (out/'per-class.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    lines=['# Training-only nested loss comparison','', 'Generated from six checksum/source/nonce-verified completed fits. Outer patients never selected their own weights, checkpoints or loss weights. Earlier development choices and same-source data limit independence. No historical final, calibration, thresholds or demo changes.','', '| Arm | Mean outer macro AUROC | Mean macro AP | Mean macro F1 | Mean micro F1 |','|---|---:|---:|---:|---:|']
    for arm,m in summary['mean_fold_metrics'].items():lines.append(f"| {arm} | {m['macro_auroc']:.5f} | {m['macro_auprc_average_precision']:.5f} | {m['macro_f1']:.5f} | {m['micro_f1']:.5f} |")
    lines+=['',f"Primary paired mean AUROC difference: {summary['mean_paired_macro_auroc_difference']:+.5f}. Candidate for future study: {summary['candidate_for_future_study']}; no deployment or independent confirmation.",'', '## All-eight assessment counts and metrics','', '| Fold | Arm | Class | + / − | AUROC | AP | Precision | Recall | F1 | TP | FP | FN |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['outer_fold']} | {r['arm']} | {r['label']} | {r['positive']} / {r['negative']} | {r['auroc']} | {r['average_precision']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['tp']} | {r['fp']} | {r['fn']} |")
    lines+=['','H rows retain sparse support, AP/recall and false negatives even if worse. See [comparison.json](comparison.json) for fold/secondary-pooled metrics and paired bootstrap intervals with defined/undefined replicate counts. Pooled scores originate from different trained models and are secondary. Intervals condition on fitted models, omit training/selection variability and are not clinical guarantees.','', '[Per-class CSV](per-class.csv) · [Prospective protocol](../../docs/NESTED_LOSS_V11.md). No final/clinical performance claim, acceptance policy, new calibrator or target accuracy is supported. All patients remain REVIEW REQUIRED.','']
    (out/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps({'status':'complete CPU comparison','candidate':summary['candidate_for_future_study']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bootstrap-batch',type=int,default=100);a=p.parse_args()
    if not 1<=a.bootstrap_batch<=2000:p.error('Batch must be 1..2000')
    main(a.bootstrap_batch)
