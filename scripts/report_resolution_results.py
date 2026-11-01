#!/usr/bin/env python3
"""Complete-matrix v13 reporting, no fitting or submissions. Fixed resumable CPU CI."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import resolution_experiment as c
from trustretina.runner.verified_artifacts import read_stage
from trustretina.data.validation import canonical_hash,file_hash
from trustretina.training.nested import patient_rows
from trustretina.training.comparison import comparison_metrics
from trustretina.evaluation.metrics import aggregate_patient_mean
from trustretina.evaluation.nested_comparison import paired_summary,bootstrap_replicate,bootstrap_intervals
ALIASES={'ratio_bce':'res224','sqrt_ratio_bce':'res320'}


def rename(value):
    # Reuse tested paired arithmetic with internal arm aliases; never imply loss changes.
    if isinstance(value,dict):return {ALIASES.get(k,k):rename(v) for k,v in value.items()}
    if isinstance(value,list):return [rename(v) for v in value]
    if isinstance(value,str):return ALIASES.get(value,value)
    return value


def main(batch):
    p=json.loads((c.BASE/'registry.json').read_text());manifest=c.verify_plan(p)
    if any(r['status']!='complete' for r in p['runs']):raise ValueError('All six completed matched fits required; no partial result/candidate')
    train=json.loads((ROOT/p['cache']['rows_path']).read_text());pairs=[{} for _ in range(3)];sources={}
    for run in p['runs']:
        report,folder,v=read_stage(ROOT/run['attempts'][-1]['bundle'])
        if report['status']!='complete' or not report.get('strict_best_reload') or not report.get('outer_assessed') or report['plan_sha256']!=p['plan_sha256'] or report['run_id']!=run['id'] or report['image_size']!=run['image_size'] or report['arm']!='ratio_bce':raise ValueError('Unfrozen/wrong resolution report')
        with np.load(folder/'outer-patients.npz',allow_pickle=False) as f:d={k:f[k] for k in ('labels','probabilities','patient_ids','component_ids')}
        if comparison_metrics(d['labels'],d['probabilities'])!=report['metrics']['patient']:raise ValueError('Patient metrics changed')
        with np.load(folder/'outer-images.npz',allow_pickle=False) as f:
            a=aggregate_patient_mean(f['patient_ids'].tolist(),f['eyes'].tolist(),f['labels'],f['probabilities'],f['component_ids'].tolist())
        if any(not np.array_equal(a[k],d[k]) for k in d):raise ValueError('Mean-eye predictor mismatch')
        expected=patient_rows([train[i] for i in manifest['folds'][run['outer_fold']]['outer_indices']])
        for k,field in [('labels','labels'),('patient_ids','patient_id'),('component_ids','component_id')]:
            if not np.array_equal(d[k],np.asarray([r[field] for r in expected])):raise ValueError('Frozen outer alignment mismatch')
        pairs[run['outer_fold']]['ratio_bce' if run['image_size']==224 else 'sqrt_ratio_bce']=d
        sources[run['id']]={'kernel_version':v['kernel_version'],'report_sha256':v['artifact_sha256'],'prediction_sha256':file_hash(folder/'outer-patients.npz'),'best_epoch':report['selection']['best_epoch']+1,'best_sha256':report['output_hashes']['best.pt'],'last_sha256':report['output_hashes']['last.pt']}
    summary=paired_summary(pairs);identity=canonical_hash({'sources':sources,'plan':p['plan_sha256']})
    target=c.BASE/'analysis-bootstrap.json';saved=json.loads(target.read_text()) if target.exists() else {'identity':identity,'replicates':[]}
    if saved['identity']!=identity:raise ValueError('Bootstrap source changed; preserve failed analysis')
    records=saved['replicates']
    for i in range(len(records),min(2000,len(records)+batch)):
        records.append(bootstrap_replicate(pairs,i,2026))
        if len(records)%20==0:c.write(target,saved)
    c.write(target,saved)
    if len(records)<2000:print(json.dumps({'completed_replicates':len(records),'required':2000,'rerun_same_command':True}));return
    summary=rename(summary);summary.update(sources=sources,plan_sha256=p['plan_sha256'],paired_component_bootstrap=bootstrap_intervals(records),
                contrast='320 minus 224, same ratio BCE',historical_final_accessed=False,calibration_fitted=False,threshold=.5,independent_confirmation=False)
    summary['warning']='Exploratory matched resolution development; conditional intervals omit training/selection variation. Fresh independent confirmation required.'
    out=ROOT/'results/resolution-v13';c.write(out/'comparison.json',summary)
    rows=[]
    for f in summary['folds']:
        for arm,m in f['arms'].items():
            for r in m['confusion']:rows.append({'fold':f['outer_fold'],'arm':arm,**m['per_class'][r['label']],**r})
    with (out/'per-class.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (out/'RESULTS.md').write_text('# Matched resolution results\n\nGenerated from all six verified fits. See comparison.json for paired means/intervals and per-class.csv for all-eight metrics, support and TP/FP/FN. No deployment, threshold fitting or independent confirmation. Negative findings must be retained.\n')
    print(json.dumps({'status':'complete CPU comparison','candidate_for_future_study':summary['candidate_for_future_study']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bootstrap-batch',type=int,default=100);a=parser.parse_args()
    if not 1<=a.bootstrap_batch<=2000:parser.error('Batch must be1..2000')
    main(a.bootstrap_batch)
