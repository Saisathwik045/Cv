#!/usr/bin/env python3
"""Audit available v13 outputs; no selection or statistics from an incomplete matrix."""
import ast,base64,csv,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import resolution_experiment as c
from trustretina.runner.verified_artifacts import read_stage
from trustretina.training.nested import loss_weights,patient_rows
from trustretina.training.controlled import advance_auc
from trustretina.training.comparison import comparison_metrics
from trustretina.evaluation.metrics import aggregate_patient_mean


def embedded_source(path,expected_hash,expected_spec):
    tree=ast.parse(Path(path).read_text());archives=[];specs=[]
    for n in ast.walk(tree):
        if not isinstance(n,ast.Call) or not isinstance(n.func,ast.Attribute) or not isinstance(n.func.value,ast.Name) or not n.args or not isinstance(n.args[0],ast.Constant):continue
        if (n.func.value.id,n.func.attr)==('base64','b64decode'):archives.append(n.args[0].value)
        if (n.func.value.id,n.func.attr)==('json','loads'):specs.append(n.args[0].value)
    if len(archives)!=1 or len(specs)!=1:raise ValueError('Unexpected submitted executable structure')
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(archives[0],validate=True))) as z:
        names=sorted(z.namelist())
        if names!=c.WORKER_FILES:raise ValueError('Actual worker archive file set differs')
        digest=hashlib.sha256(b''.join(n.encode()+z.read(n) for n in names)).hexdigest()
    if digest!=expected_hash:raise ValueError('Actual embedded worker differs from approved source')
    spec=json.loads(specs[0])
    if any(spec.get(k)!=v for k,v in expected_spec.items()):raise ValueError('Actual submitted spec differs from frozen approval')
    return digest


def inner_selection(report):
    state={}
    for r in report['history']:
        if any(k.startswith('outer_') for k in r):raise ValueError('Outer metrics present during checkpoint selection')
        if r['inner_patient_macro_auroc']!=r['inner_patient_metrics']['macro_auroc']:raise ValueError('Inner metric mismatch')
        state,_=advance_auc(state,r['inner_patient_macro_auroc'],r['epoch']-1,min_delta=1e-4,patience=6)
    if state!=report['selection']:raise ValueError('Inner-only selection replay differs')
    return {'epochs':len(report['history']),'best_epoch':state['best_epoch']+1,'inner_only_selection_reproduced':True}


def main():
    p=json.loads((c.BASE/'registry.json').read_text());manifest=c.verify_plan(p);rows=json.loads((ROOT/p['cache']['rows_path']).read_text())
    audits=[];classes=[];sources=[]
    for run in p['runs']:
        for attempt in run['attempts']:
            if attempt.get('status') not in ('complete','paused','failed'):continue
            bundle=ROOT/attempt['bundle'];report,folder,v=read_stage(bundle)
            if report['plan_sha256']!=p['plan_sha256'] or report['run_id']!=run['id'] or report['image_size']!=run['image_size']:raise ValueError('Wrong report identity')
            meta=json.loads((bundle/'remote-source/kernel-metadata.json').read_text())
            actual=embedded_source(bundle/'remote-source'/meta['code_file'],p['experiment_source_sha256'],{
              'config':p['config'],'config_sha256':p['config_sha256'],'run_id':run['id'],'image_size':run['image_size'],'outer_fold':run['outer_fold'],
              'fold_manifest':manifest,'plan_sha256':p['plan_sha256'],'inputs_sha256':p['cache']['inputs_sha256'],'dataset_sources':[]})
            sources.append({'run':run['id'],'version':v['kernel_version'],'source_sha256':v['source_sha256'],'embedded_worker_sha256':actual,
                            'report_sha256':v['artifact_sha256'],'output_hashes':report['output_hashes'],'dependencies':report.get('dependencies'),
                            'worker_seconds':report['elapsed_seconds'],'gpu':report.get('gpu'),'status':report['status']})
            row={'run':run['id'],'resolution':run['image_size'],'status':report['status'],'version':v['kernel_version']}
            if report.get('history'):row.update(inner_selection(report))
            if report.get('positive_weights'):
                fit=[rows[i] for i in manifest['folds'][run['outer_fold']]['fit_indices']]
                np.testing.assert_allclose(report['positive_weights'],loss_weights(fit,'ratio_bce'),rtol=0,atol=1e-12)
            if report['status']=='paused' and report.get('outer_assessed'):raise ValueError('Paused fit accessed outer patients')
            if report['status']=='complete':
                if not report.get('strict_best_reload') or not report.get('outer_assessed'):raise ValueError('Outer freeze/reload missing')
                with np.load(folder/'outer-patients.npz',allow_pickle=False) as f:d={k:f[k] for k in ('labels','probabilities','patient_ids','component_ids')}
                m=comparison_metrics(d['labels'],d['probabilities'])
                if m!=report['metrics']['patient']:raise ValueError('Patient metrics cannot be reproduced')
                with np.load(folder/'outer-images.npz',allow_pickle=False) as f:
                    a=aggregate_patient_mean(f['patient_ids'].tolist(),f['eyes'].tolist(),f['labels'],f['probabilities'],f['component_ids'].tolist())
                if any(not np.array_equal(a[k],d[k]) for k in d):raise ValueError('Mean-eye aggregation cannot be reproduced')
                expected=patient_rows([rows[i] for i in manifest['folds'][run['outer_fold']]['outer_indices']])
                for key,field in [('patient_ids','patient_id'),('labels','labels'),('component_ids','component_id')]:
                    if not np.array_equal(d[key],np.asarray([r[field] for r in expected])):raise ValueError('Wrong frozen outer patient/label/group alignment')
                row.update(macro_auroc=m['macro_auroc'],macro_ap=m['macro_auprc_average_precision'],macro_f1=m['macro_f1'],micro_f1=m['micro_f1'])
                for label,x in m['per_class'].items():
                    pred=d['probabilities'][:,list('NDGCAHMO').index(label)]>=.5;y=d['labels'][:,list('NDGCAHMO').index(label)].astype(bool)
                    classes.append({'run':run['id'],'resolution':run['image_size'],'label':label,**x,'tp':int((pred&y).sum()),'fp':int((pred&~y).sum()),'fn':int((~pred&y).sum())})
            audits.append(row)
    complete=all(r['status']=='complete' for r in p['runs'])
    result={'plan_sha256':p['plan_sha256'],'complete_matrix':complete,'selected_candidate':None,'selection_note':'This execution audit never selects a candidate; only the frozen complete-matrix reporter may do so.',
            'runs':audits,'artifacts':sources,'per_class':classes,'historical_final_accessed':False,'calibration_fitted':False,'demo_replaced':False}
    out=ROOT/'results/resolution-v13';c.write(out/'execution-audit.json',result)
    if classes:
        with (out/'executed-per-class.csv').open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=list(classes[0]),lineterminator='\n');w.writeheader();w.writerows(classes)
    lines=['# V13 verified execution audit','', 'All six completed; use the separately generated frozen paired comparison.' if complete else '**Incomplete matrix: no winner or paired conclusion.**','', '| Run | Status | Epochs / best | Macro AUROC | Macro AP | Macro F1 | Micro F1 |','|---|---|---:|---:|---:|---:|---:|']
    for r in audits:
        lines.append(f"| {r['run']} | {r['status']} | {r.get('epochs','—')} / {r.get('best_epoch','—')} | {r.get('macro_auroc','—')} | {r.get('macro_ap','—')} | {r.get('macro_f1','—')} | {r.get('micro_f1','—')} |")
    lines+=['','Source/nonce/output hashes checked; embedded worker statically decoded without execution and matched approved bytes. Fit-only ratio weights, inner-only checkpoint selection, frozen outer labels/groups and mean-eye probabilities are verified for completed outputs. Patient-level artifacts stay private. No model promotion, final evaluation, calibration, threshold fitting or all-data refit.','']
    (out/'EXECUTION.md').write_text('\n'.join(lines));print(json.dumps({'complete_matrix':complete,'verified_attempts':len(audits),'selected_candidate':None}))


if __name__=='__main__':main()
