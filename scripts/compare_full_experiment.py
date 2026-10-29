#!/usr/bin/env python3
"""Generate aggregate tables and plots from verified best-checkpoint NPZ predictions."""
import argparse,csv,json,os,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trustretina.runner.verified_artifacts import read_stage
from trustretina.data.validation import file_hash
from trustretina.training.comparison import comparison_metrics,select_backbone
from trustretina.evaluation.metrics import aggregate_patient_mean


def compare(base,output, *, update_registry=True):
    registry=json.loads((base/'registry.json').read_text());reports={};reference=None;runtime=None;rows=[];histories={}
    index=ROOT/registry['cache']['bundle']/'download/prepared/inputs.json'
    if file_hash(index)!=registry['cache']['inputs_sha256']:raise ValueError('Frozen comparison input index changed')
    cache=json.loads(index.read_text());v=cache['roles']['validation'];rowpath=index.parent/v['rows_file']
    if file_hash(rowpath)!=v['rows_sha256']:raise ValueError('Frozen validation rows changed')
    frozen=json.loads(rowpath.read_text())
    expected_keys=[(r['patient_id'],r['eye'],r['path'],r['component_id']) for r in frozen]
    expected_labels=np.asarray([r['labels'] for r in frozen])
    for run in registry['runs']:

        if run['status']!='complete':continue
        bundle=ROOT/run['attempts'][-1]['bundle'];report,folder,_=read_stage(bundle)
        if report['status']!='complete' or report['release_sha256']!=registry['config']['release_manifest_sha256'] or report['config_sha256']!=registry['config_sha256']:raise ValueError('Comparison source/config mismatch')
        state=json.loads((folder/'progress.json').read_text())
        if state['provenance']!=report['provenance'] or state['selection']!=report['selection']:raise ValueError('Checkpoint/report provenance mismatch')
        for filename in ('best.pt','last.pt','history.json'):
            if file_hash(folder/filename)!=state['hashes'][filename]:raise ValueError('Checkpoint/progress checksum mismatch')
        for filename in ('best.pt','last.pt'):
            marker=folder/(filename+'.sha256')
            if marker.exists() and marker.read_text().strip()!=state['hashes'][filename]:raise ValueError('Checkpoint marker mismatch')
        if runtime is None:runtime=report['dependencies']
        elif runtime!=report['dependencies']:raise ValueError('Mixed runtime environments; comparison requires review')
        with np.load(folder/'validation-images.npz',allow_pickle=False) as eye:
            data={k:eye[k] for k in eye.files}
        keys=list(zip(data['patient_ids'],data['eyes'],data['image_ids'],data['component_ids']))
        if keys!=expected_keys or not np.array_equal(data['labels'],expected_labels):raise ValueError('Prediction alignment differs from frozen validation')
        if len(set(keys))!=len(keys):raise ValueError('Repeated validation sample')
        if reference is None:reference=(keys,data['labels'])
        elif keys!=reference[0] or not np.array_equal(data['labels'],reference[1]):raise ValueError('Validation sample alignment differs across models')
        patient=aggregate_patient_mean(data['patient_ids'].tolist(),data['eyes'].tolist(),data['labels'],data['probabilities'],data['component_ids'].tolist())
        with np.load(folder/'validation-patients.npz',allow_pickle=False) as p:
            if p['patient_ids'].tolist()!=patient['patient_ids'] or not np.array_equal(p['labels'],patient['labels']) or not np.array_equal(p['probabilities'],patient['probabilities']):raise ValueError('Patient prediction artifact disagrees with eye aggregation')
        metrics={'image':comparison_metrics(data['labels'],data['probabilities']),'patient':comparison_metrics(patient['labels'],patient['probabilities'])}
        reports[run['backbone']]=metrics;histories[run['backbone']]=json.loads((folder/'history.json').read_text())
        m=metrics['patient'];rows.append({'backbone':run['backbone'],'patient_macro_auroc':m['macro_auroc'],'patient_macro_auprc':m['macro_auprc_average_precision'],'patient_macro_f1':m['macro_f1'],'patient_micro_f1':m['micro_f1'],'patient_subset_accuracy':m['subset_exact_match_accuracy'],'patient_labelwise_accuracy':m['labelwise_binary_accuracy'],'parameters':report['parameter_count'],'epochs':report['completed_epochs'],'best_epoch':report['selection']['best_epoch']+1,'worker_seconds':report['elapsed_seconds'],'inference_ms_per_image':report['metrics']['inference']['milliseconds_per_image']})
    output.mkdir(parents=True,exist_ok=True)
    selected=select_backbone(registry['config']['backbones'],reports)
    result={'completed_models':len(reports),'selected_backbone':selected,'primary_criterion':'validation patient macro AUROC of minimum patient-validation-loss checkpoint; all five required','threshold':.5,'runtime_dependencies':runtime,'reports':reports,'comparison':rows,'calibration_accessed':False,'final_evaluation_accessed':False}
    (output/'comparison.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    lines=['# Validation development comparison','',
           'Generated from checksum-verified best-checkpoint prediction files. Patient probabilities are the mean of both eyes; threshold p >= 0.5. This is development evidence, not final evaluation.','',
           '| Backbone | State | Epochs / best | Patient macro AUROC | Macro AP | Macro F1 | Micro F1 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for run in registry['runs']:
        values=next((x for x in rows if x['backbone']==run['backbone']),None)
        if values:
            lines.append('| '+run['backbone']+' | complete | '+str(values['epochs'])+' / '+str(values['best_epoch'])+' | '+' | '.join('undefined' if values[k] is None else f'{values[k]:.4f}' for k in ('patient_macro_auroc','patient_macro_auprc','patient_macro_f1','patient_micro_f1'))+' |')
        else:lines.append('| '+run['backbone']+' | '+run['status']+' | — | — | — | — | — |')
    lines+=['','Selected backbone: '+(selected or 'pending completion/defined rankings for all five')+'.','']
    if reports:
        first=next(iter(reports.values()))['patient']
        lines+=['| Label | Validation positive / negative patients |','|---|---:|']
        for label in first['class_order']:
            support=first['per_class'][label];lines.append(f"| {label} | {support['positive']} / {support['negative']} |")
        lines+=['','H has only 16 positive validation patients; class-specific estimates are unstable. No calibration or acceptance-risk claims.','',
                '[Per-class and image/patient metrics](comparison.json) · [CSV](comparison.csv) · [AUROC plot](validation-auroc.png) · [Loss curves](validation-loss.png)','']
    (output/'README.md').write_text('\n'.join(lines))
    if rows:
        with (output/'comparison.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'trustretina-mpl'))
        import matplotlib;matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig,ax=plt.subplots(figsize=(10,4));ax.bar([r['backbone'] for r in rows],[r['patient_macro_auroc'] or 0 for r in rows]);ax.set_ylim(0,1);ax.set_ylabel('Validation patient macro AUROC');ax.set_title('Development selection; completed models only');fig.tight_layout();fig.savefig(output/'validation-auroc.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4))
        for name,history in histories.items():ax.plot([r['epoch'] for r in history],[r['validation_patient_weighted_bce'] for r in history],label=name)
        ax.set_xlabel('Completed epoch');ax.set_ylabel('Patient validation weighted BCE');ax.legend();fig.tight_layout();fig.savefig(output/'validation-loss.png',dpi=160);plt.close(fig)
    if update_registry:
        registry['selected_backbone']=selected
        # Single controller lock prevents racing an advance/export.
        temporary=base/'registry.tmp';temporary.write_text(json.dumps(registry,indent=2,sort_keys=True));temporary.replace(base/'registry.json')
    print(json.dumps({'completed_models':len(reports),'selected_backbone':selected}))

if __name__=='__main__':
    import fcntl
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=ROOT/'artifacts/runner/full-v5');p.add_argument('--output',type=Path,default=ROOT/'results/full-v5');p.add_argument('--snapshot',action='store_true',help='Read atomic registry without updating it while a driver runs');args=p.parse_args()
    if args.snapshot:compare(args.base,args.output,update_registry=False)
    else:
        with (args.base/'controller.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);compare(args.base,args.output)
