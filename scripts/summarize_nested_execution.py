#!/usr/bin/env python3
"""Verify available v11 artifacts, including incomplete matrices; never choose a partial winner."""
import ast,base64,csv,hashlib,io,json,os,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import nested_experiment as c
from trustretina.runner.verified_artifacts import read_stage
from trustretina.training.nested import patient_rows,loss_weights
from trustretina.training.controlled import advance_auc
from trustretina.training.comparison import comparison_metrics
from trustretina.evaluation.metrics import aggregate_patient_mean
from trustretina.evaluation.threshold_cv import summarize
from report_controlled_results import verify_history_continuation


def verify_embedded_worker(path,expected_source,expected_spec):
    tree=ast.parse(Path(path).read_text())
    def strings(owner,method):
        return [n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
                and n.func.value.id==owner and n.func.attr==method and n.args
                and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)]
    archives=strings('base64','b64decode');specs=strings('json','loads')
    if len(archives)!=1 or len(specs)!=1:raise ValueError('Remote executable has unexpected source/spec structure')
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(archives[0],validate=True))) as z:
        names=sorted(n for n in z.namelist() if n.startswith('src/') and n.endswith('.py'))
        digest=hashlib.sha256(b''.join(n.encode()+z.read(n) for n in names)).hexdigest()
    if digest!=expected_source:raise ValueError('Actual embedded worker bytes differ from approved source hash')
    actual=json.loads(specs[0])
    if any(actual.get(k)!=v for k,v in expected_spec.items()):raise ValueError('Actual submitted experiment spec differs from frozen identity')
    return digest


def audit_inner_history(report):
    selection={}
    for entry in report['history']:
        if any(key.startswith('outer_') for key in entry):raise ValueError('Epoch history contains outer assessment data')
        if entry['inner_patient_macro_auroc']!=entry['inner_patient_metrics']['macro_auroc']:raise ValueError('Inner selection diagnostic mismatch')
        selection,_=advance_auc(selection,entry['inner_patient_macro_auroc'],entry['epoch']-1,min_delta=1e-4,patience=6)
    if selection!=report['selection']:raise ValueError('Best/stop state does not reproduce from inner history only')
    return {'best_epoch':selection['best_epoch']+1,'epochs':len(report['history']),'inner_only_selection_reproduced':True}


def run():
    p=json.loads((c.BASE/'registry.json').read_text());manifest=c.verify_plan(p)
    training=json.loads((ROOT/p['cache']['training_metadata_file']).read_text())
    complete=all(r['status']=='complete' for r in p['runs']);runs=[];classes=[];artifacts=[];history_checks=[];histories={}
    for r in p['runs']:
        row={'id':r['id'],'outer_fold':r['outer_fold'],'arm':r['arm'],'status':r['status'],
             'versions':[a.get('kernel_version',a['kernel_id']) for a in r['attempts']]}
        for index,attempt in enumerate(r['attempts']):
            if attempt.get('status') not in ('complete','paused','failed'):continue
            report,folder,v=read_stage(ROOT/attempt['bundle'])
            if report.get('plan_sha256')!=p['plan_sha256'] or report.get('run_id')!=r['id']:raise ValueError('Artifact belongs to another plan/run')
            metadata=json.loads((ROOT/attempt['bundle']/'remote-source/kernel-metadata.json').read_text())
            embedded=verify_embedded_worker(ROOT/attempt['bundle']/'remote-source'/metadata['code_file'],p['experiment_source_sha256'],{
                'config':p['config'],'config_sha256':p['config_sha256'],'plan_sha256':p['plan_sha256'],
                'run_id':r['id'],'arm':r['arm'],'outer_fold':r['outer_fold'],'fold_manifest':manifest,
                'fold_manifest_sha256':p['fold_manifest_sha256'],'dataset_sources':[]})
            if report.get('provenance'):
                if report['provenance']['fold_manifest_sha256']!=p['fold_manifest_sha256'] or report['provenance']['experiment_source_sha256']!=p['experiment_source_sha256']:raise ValueError('Wrong worker/fold provenance')
                fit=[training[i] for i in manifest['folds'][r['outer_fold']]['fit_indices']]
                np.testing.assert_allclose(report['positive_weights'],loss_weights(fit,r['arm']),rtol=0,atol=1e-12)
            artifacts.append({'run':r['id'],'kernel_version':v['kernel_version'],'status':report['status'],'source_sha256':v['source_sha256'],
                              'report_sha256':v['artifact_sha256'],'actual_embedded_worker_sha256':embedded,'output_sha256':report['output_hashes'],'worker_elapsed_seconds':report['elapsed_seconds'],
                              'gpu':report.get('gpu'),'dependencies':report.get('dependencies')})
            if report.get('history'):
                check=audit_inner_history(report);history_checks.append({'run':r['id'],'version':v['kernel_version'],**check})
            if index and report.get('history'):
                _,old,_=read_stage(ROOT/r['attempts'][index-1]['bundle'])
                verify_history_continuation(json.loads((old/'history.json').read_text()),report['history'])
            if report['status']=='paused' and report.get('outer_assessed'):raise ValueError('Paused session exposed outer data')
            if report['status']!='complete':continue
            if not report.get('strict_best_reload') or not report.get('outer_assessed'):raise ValueError('Outer checkpoint not frozen/reloaded')
            if report['arm']!=r['arm'] or report['outer_fold']!=r['outer_fold']:raise ValueError('Wrong arm/fold')
            with np.load(folder/'outer-patients.npz',allow_pickle=False) as f:data={k:f[k] for k in ('labels','probabilities','patient_ids','component_ids')}
            metrics=comparison_metrics(data['labels'],data['probabilities'])
            if metrics!=report['metrics']['patient']:raise ValueError('Outer metrics do not reproduce')
            with np.load(folder/'outer-images.npz',allow_pickle=False) as f:
                if comparison_metrics(f['labels'],f['probabilities'])!=report['metrics']['image']:raise ValueError('Image metrics do not reproduce')
                patient=aggregate_patient_mean(f['patient_ids'].tolist(),f['eyes'].tolist(),f['labels'],f['probabilities'],f['component_ids'].tolist())
            if any(not np.array_equal(patient[key],data[key]) for key in data):raise ValueError('Saved mean-eye aggregation mismatch')
            expected=patient_rows([training[i] for i in manifest['folds'][r['outer_fold']]['outer_indices']])
            for key,field in [('patient_ids','patient_id'),('component_ids','component_id'),('labels','labels')]:
                if not np.array_equal(data[key],np.asarray([a[field] for a in expected])):raise ValueError('Predictions differ from prespecified outer patients')
            confusion=summarize(data['labels'],data['probabilities']>=.5)
            row.update(best_epoch=report['selection']['best_epoch']+1,epochs=report['completed_epochs'],
                       macro_auroc=metrics['macro_auroc'],macro_ap=metrics['macro_auprc_average_precision'],
                       macro_f1=metrics['macro_f1'],micro_f1=metrics['micro_f1'],
                       total_tp=confusion['total_tp'],total_fp=confusion['total_fp'],total_fn=confusion['total_fn'])
            for item in confusion['per_class']:
                ranking=metrics['per_class'][item['label']]
                classes.append({'run':r['id'],'outer_fold':r['outer_fold'],'arm':r['arm'],**item,'auroc':ranking['auroc'],'average_precision':ranking['average_precision']})
            histories[r['id']]=report['history']
        runs.append(row)
    result={'plan_sha256':p['plan_sha256'],'complete_matrix':complete,'selected_candidate':None,
            'selection_note':'No selection in this execution audit. Only the prespecified complete-matrix comparison can select a future candidate.',
            'runs':runs,'per_class':classes,'artifact_inventory':artifacts,'inner_checkpoint_checks':history_checks,
            'charged_quota_seconds_last_observed':p.get('charged_quota_seconds_observed'),'reserved_budget_seconds':sum(a['reserved_seconds'] for r in p['runs'] for a in r['attempts']),
            'gpu_submission_count':sum(len(r['attempts']) for r in p['runs']),'maximum_gpu_quota_seconds':14400,
            'independent_confirmation':False,'demo_replaced':False,'historical_final_opened':False,'calibration_fitted':False,'threshold':.5}
    out=ROOT/'results/nested-v11';c.write(out/'execution-audit.json',result)
    if classes:
        with (out/'executed-per-class.csv').open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=list(classes[0]),lineterminator='\n');w.writeheader();w.writerows(classes)
    lines=['# V11 execution and artifact audit','',('All six fits completed. See the separately generated prespecified paired comparison; this artifact audit does not perform selection.' if complete else '**Incomplete matrix: no winner or overall paired conclusion is declared.**'),'',
           'Each published metric is recomputed from checksum/source/nonce-verified outer prediction files. Patient IDs, labels and containment groups match the frozen training-only outer assignments; mean-eye aggregation is reconstructed from saved eye probabilities. Best/stop state is replayed from inner history only. Patient-level artifacts stay private.','',
           '| Run | Status | Epochs / best | Outer macro AUROC | Macro AP | Macro F1 | Micro F1 | TP / FP / FN |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        if 'macro_auroc' not in r:lines.append(f"| {r['id']} | {r['status']} | — | — | — | — | — | — |")
        else:lines.append(f"| {r['id']} | {r['status']} | {r['epochs']} / {r['best_epoch']} | {r['macro_auroc']:.5f} | {r['macro_ap']:.5f} | {r['macro_f1']:.5f} | {r['micro_f1']:.5f} | {r['total_tp']} / {r['total_fp']} / {r['total_fn']} |")
    lines+=['','## All-eight completed-run counts','', '| Run | Class | + / − | AUROC | AP | Precision | Recall | F1 | TP | FP | FN |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in classes:lines.append(f"| {r['run']} | {r['label']} | {r['positive']} / {r['negative']} | {r['auroc']:.4f} | {r['average_precision']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['tp']} | {r['fp']} | {r['fn']} |")
    lines+=['','GPU-quota totals come from saved API counters, not worker times. Per-session quota snapshots and receipts are private, with requested accelerator, timeout and source/checkpoint/output hashes preserved. The final global counter delta is authoritative; account intervals can include delayed billing.','',
            'No full-data refit, old calibration temperature, threshold tuning, acceptance policy or demo promotion occurs. Same-source data, earlier adaptive choices, sparse H and one seed still limit independence. A fresh suitable independent cohort is required for new performance claims. REVIEW REQUIRED remains unchanged.','',
            '[Machine-readable audit and hashes](execution-audit.json) · [Per-class CSV](executed-per-class.csv) · [Prospective protocol](../../docs/NESTED_LOSS_V11.md)','']
    (out/'EXECUTION.md').write_text('\n'.join(lines))
    if histories:plot(histories,out)
    print(json.dumps({'complete_matrix':complete,'completed_runs':len([r for r in runs if r['status']=='complete']),'selected_candidate':None}))


def plot(histories,out):
    os.environ.setdefault('MPLCONFIGDIR',str(c.BASE/'mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name,h in histories.items():
        k=int(name[4]);axes[k].plot([r['epoch'] for r in h],[r['inner_patient_macro_auroc'] for r in h],marker='o',label=name)
    for k,ax in enumerate(axes):
        ax.set_title(f'Fold {k}: inner selection AUROC');ax.set_xlabel('Epoch');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'inner-learning-curves.png',dpi=150);plt.close(fig)


if __name__=='__main__':run()
