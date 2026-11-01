#!/usr/bin/env python3
"""Prediction-derived development comparison. Never opens calibration/final arrays."""
import argparse,csv,json,math,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trustretina.config import CLASSES
from trustretina.training.comparison import comparison_metrics
from trustretina.evaluation.metrics import aggregate_patient_mean
from trustretina.runner.verified_artifacts import read_stage
from trustretina.data.validation import file_hash
OUT=ROOT/'results/round2-v9'


def confusion_rows(y,p):
    y=np.asarray(y);p=np.asarray(p)
    if y.ndim!=2 or y.shape!=p.shape or y.shape[1]!=8 or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('Aligned finite binary (n,8) arrays required')
    rows=[];pred=p>=.5
    for j,c in enumerate(CLASSES):
        a=y[:,j]==1;b=pred[:,j];tp=int((a&b).sum());fp=int((~a&b).sum());fn=int((a&~b).sum());tn=int((~a&~b).sum());positive=tp+fn;negative=fp+tn
        rows.append({'label':c,'positive':positive,'negative':negative,'true_positive':tp,'false_positive':fp,'false_negative':fn,'true_negative':tn,'predicted_positive':tp+fp,'precision_defined':bool(tp+fp),'recall_defined':bool(positive),'f1_defined':bool(2*tp+fp+fn),'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/positive if positive else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'false_positive_rate':fp/negative if negative else None,'false_negative_rate':fn/positive if positive else None})
    return rows


def read_predictions(bundle):
    report,folder,verification=read_stage(bundle)
    if report['status']!='complete' or not report['strict_best_reload']:raise ValueError('Only completed and strictly reloaded jobs may enter comparison')
    with np.load(folder/'validation-patients.npz',allow_pickle=False) as f:
        d={k:f[k] for k in ('labels','probabilities','patient_ids','component_ids')}
    metrics=comparison_metrics(d['labels'],d['probabilities'])
    if metrics!=report['metrics']['patient']:raise ValueError('Stored validation metrics do not reproduce')
    with np.load(folder/'validation-images.npz',allow_pickle=False) as f:
        images={k:f[k] for k in ('labels','probabilities','patient_ids','eyes','component_ids')}
    image_metrics=comparison_metrics(images['labels'],images['probabilities'])
    if image_metrics!=report['metrics']['image']:raise ValueError('Image metrics do not reproduce')
    aggregate=aggregate_patient_mean(images['patient_ids'].tolist(),images['eyes'].tolist(),images['labels'],images['probabilities'],images['component_ids'].tolist())
    for key in ('labels','probabilities','patient_ids','component_ids'):
        if not np.array_equal(np.asarray(aggregate[key]),d[key]):raise ValueError('Stored patient predictor differs from declared mean-eye aggregation')
    per_class=confusion_rows(d['labels'],d['probabilities'])
    for row in per_class:
        stored=metrics['per_class'][row['label']]
        for key in ('precision','recall','f1'):
            if not math.isclose(row[key],stored[key],rel_tol=0,abs_tol=1e-12):raise ValueError('Confusion/metric disagreement')
        row.update(auroc=stored['auroc'],average_precision=stored['average_precision'])
    return report,folder,verification,d,metrics,per_class,image_metrics


def verify_history_continuation(previous,current):
    if current[:len(previous)]!=previous or len(current)<=len(previous):raise ValueError('Resume did not preserve and extend the prior epoch history')
    old=previous[-1];new=current[len(previous)]
    if new['epoch']!=old['epoch']+1 or new['global_step']<=old['global_step']:raise ValueError('Resume repeated/skipped a completed epoch boundary')
    if not math.isclose(new['learning_rate'],old['next_learning_rate'],rel_tol=0,abs_tol=1e-15):raise ValueError('Resume learning rate differs from saved scheduler state')


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)


def run(*,allow_incomplete=False):
    registry=json.loads((ROOT/'artifacts/runner/round2-v9/registry.json').read_text())
    import controlled_experiment as controller
    controller.verify_plan(registry)
    old=json.loads((ROOT/'artifacts/runner/full-v5/registry.json').read_text());complete=all(r['status']=='complete' for r in registry['runs'])
    if not complete and not allow_incomplete:raise ValueError('Four completed runs required; --allow-incomplete labels a partial report and never selects a winner')
    summary=[];class_rows=[];sources={};reference=None;baseline={};histories={};image_reports={}
    def add(name,kind,backbone,schedule,bundle,worker_seconds):
        nonlocal reference
        report,folder,v,d,m,classes,image_metrics=read_predictions(bundle)
        image_reports[name]=image_metrics
        if report['backbone']!=backbone:raise ValueError('Wrong backbone source')
        if kind=='controlled_v9' and (report.get('plan_sha256')!=registry['plan_sha256'] or report.get('run_id')!=name or report['config_sha256']!=registry['config_sha256']):raise ValueError('Controlled run/protocol identity differs')
        if reference is None:reference=d
        elif any(not np.array_equal(d[k],reference[k]) for k in ('labels','patient_ids','component_ids')):raise ValueError('Patient/component/label alignment differs from original validation baseline')
        if len(d['labels'])!=525:raise ValueError('Expected all 525 frozen validation patients')
        row={'run':name,'kind':kind,'backbone':backbone,'schedule':schedule,'epochs_completed':report['completed_epochs'],'best_epoch':report['selection']['best_epoch']+1,'patient_macro_auroc':m['macro_auroc'],'patient_macro_ap':m['macro_auprc_average_precision'],'patient_macro_f1':m['macro_f1'],'patient_micro_f1':m['micro_f1'],'patient_subset_exact_match':m['subset_exact_match_accuracy'],'patient_labelwise_binary_accuracy':m['labelwise_binary_accuracy'],'patient_hamming_loss':m['hamming_loss'],'false_positive_label_decisions':sum(r['false_positive'] for r in classes),'false_negative_label_decisions':sum(r['false_negative'] for r in classes),'worker_seconds':worker_seconds,'best_checkpoint_sha256':report['output_hashes']['best.pt'],'last_checkpoint_sha256':report['output_hashes']['last.pt']}
        summary.append(row);sources[name]={'kernel_version':v['kernel_version'],'report_sha256':v['artifact_sha256'],'patient_predictions_sha256':file_hash(folder/'validation-patients.npz'),'best_sha256':row['best_checkpoint_sha256'],'last_sha256':row['last_checkpoint_sha256'],'history_sha256':file_hash(folder/'history.json')};histories[name]=report['history']
        if kind=='original_v5':baseline[backbone]=(row,{r['label']:r for r in classes})
        for r in classes:
            b=baseline[backbone][1][r['label']]
            class_rows.append({'run':name,'kind':kind,'backbone':backbone,**r,'delta_fp_vs_original_backbone':r['false_positive']-b['false_positive'],'delta_f1_vs_original_backbone':r['f1']-b['f1']})
    for name in ('convnext_tiny','densenet121'):
        entry=next(r for r in old['runs'] if r['backbone']==name)
        add('original_'+name,'original_v5',name,'constant',ROOT/entry['attempts'][-1]['bundle'],entry['elapsed_seconds'])
    resume_checks=[]
    for r in registry['runs']:
        if r['status']=='complete' and len(r['attempts'])>1:
            for prior,later in zip(r['attempts'],r['attempts'][1:]):
                _,a,_=read_stage(ROOT/prior['bundle']);_,b,_=read_stage(ROOT/later['bundle'])
                old_history=json.loads((a/'history.json').read_text());new_history=json.loads((b/'history.json').read_text())
                verify_history_continuation(old_history,new_history)
                resume_checks.append({'run':r['id'],'previous_version':prior['kernel_version'],'continued_version':later['kernel_version'],'completed_epochs_before_resume':len(old_history),'first_resumed_epoch':new_history[len(old_history)]['epoch'],'history_prefix_and_learning_rate_verified':True})
        if r['status']=='complete':add(r['id'],'controlled_v9',r['backbone'],r['schedule'],ROOT/r['attempts'][-1]['bundle'],r['total_worker_seconds'])
    for row in summary:
        base=baseline[row['backbone']][0]
        row['delta_auroc_vs_original_backbone']=row['patient_macro_auroc']-base['patient_macro_auroc'];row['delta_macro_f1_vs_original_backbone']=row['patient_macro_f1']-base['patient_macro_f1']
    original_selected=baseline[old['selected_backbone']]
    for row in summary:
        row['delta_auroc_vs_original_selected']=row['patient_macro_auroc']-original_selected[0]['patient_macro_auroc']
        row['delta_macro_f1_vs_original_selected']=row['patient_macro_f1']-original_selected[0]['patient_macro_f1']
    for row in class_rows:
        b=original_selected[1][row['label']]
        row['delta_fp_vs_original_selected']=row['false_positive']-b['false_positive']
        row['delta_f1_vs_original_selected']=row['f1']-b['f1']
    new=[r for r in summary if r['kind']=='controlled_v9'];winner=max(new,key=lambda r:r['patient_macro_auroc'])['run'] if complete else None
    result={'scope':'Repeated-development validation only; no final/calibration evaluation or tuning','plan_sha256':registry['plan_sha256'],'complete_matrix':complete,'selected_run':winner,'patients':525,'components':len(set(reference['component_ids'].tolist())),'threshold':.5,'patient_aggregation':'mean left/right eye sigmoid probabilities','comparison':summary,'image_metrics':image_reports,'per_class':class_rows,'sources':sources,'resume_checks':resume_checks,'job_states':[{k:r[k] for k in ('id','backbone','schedule','status')} for r in registry['runs']],'reserved_seconds':sum(a['reserved_seconds'] for r in registry['runs'] for a in r['attempts']),'demo_replaced':False,'calibration_fitted':False,'historical_final_used':False,'limitations':['Reused development cohort; checkpoint/backbone selection optimism','One training seed; no estimate of seed variance','Rare H has 16 validation positives','Labels are patient-derived, not independently verified eye-specific diagnoses','V9 vs v5 also changes checkpoint/early-stop/budget rules; schedule-only contrast is within each v9 pair','New independent suitable evaluation cohort required for a new performance claim','All patients remain REVIEW REQUIRED']}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'verified-comparison.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False));write_csv(OUT/'validation-comparison.csv',summary);write_csv(OUT/'per-class-false-positives-f1.csv',class_rows)
    lines=['# Controlled second-round validation results','', 'Generated from retrieved checksum/source/nonce-verified prediction files. All comparisons use the same 525 patients, 518 containment components, eight labels and fixed p ≥ 0.5. No historical final or calibration prediction arrays were opened.','',('All four runs complete; primary selected development run: **'+winner+'**.' if complete else '**Partial matrix. No winner selected.**'),'', '| Run | Epochs / best | Macro AUROC | Macro AP | Macro F1 | Micro F1 | Δ AUROC vs original selected | Δ macro F1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['run']} | {r['epochs_completed']} / {r['best_epoch']} | {r['patient_macro_auroc']:.5f} | {r['patient_macro_ap']:.5f} | {r['patient_macro_f1']:.5f} | {r['patient_micro_f1']:.5f} | {r['delta_auroc_vs_original_selected']:+.5f} | {r['delta_macro_f1_vs_original_selected']:+.5f} |")
    lines+=['','![Completed validation histories](controlled-learning-curves.png)','','Checkpoint/backbone selection is macro AUROC, not F1. The old v5 checkpoints were selected by validation loss. Thus v9-versus-v5 comparisons also change checkpoint/early-stop rules and training budget; only the within-v9 same-backbone constant/cosine contrast changes the LR schedule alone.','', '## All-eight false positives and F1','', '| Run | Label | Positive / negative | TP | FP | FN | Precision | Recall | F1 | Δ FP vs original selected | Δ F1 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in class_rows:lines.append(f"| {r['run']} | {r['label']} | {r['positive']} / {r['negative']} | {r['true_positive']} | {r['false_positive']} | {r['false_negative']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['delta_fp_vs_original_selected']:+d} | {r['delta_f1_vs_original_selected']:+.4f} |")
    lines+=['','[Full permitted metrics/provenance](verified-comparison.json) · [Comparison CSV](validation-comparison.csv) · [Per-class FP/FN/F1 CSV](per-class-false-positives-f1.csv)','', '## Accuracy definitions and limitations','', 'Exact match requires all eight labels correct. Labelwise binary accuracy averages correct label bits; it is not diagnostic accuracy. Hamming loss is its complement. Macro F1 averages eight F1 values; micro F1 uses pooled TP/FP/FN on the (n,8) matrix. Rare support and undefined-denominator conventions remain visible in the artifacts.','',*['- '+x for x in result['limitations']],'', 'No result changes the model, thresholds, calibration, splits or pipeline after seeing this development comparison. No demo promotion or automatic ACCEPT policy occurs.','']
    if len(new):
        lines+=['## Observed metric tradeoffs','']
        for row in new:
            classes=[x for x in class_rows if x['run']==row['run']];up=[x['label'] for x in classes if x['delta_fp_vs_original_selected']>0];down=[x['label'] for x in classes if x['delta_f1_vs_original_selected']<0]
            lines.append(f"- {row['run']}: Δ macro AUROC {row['delta_auroc_vs_original_selected']:+.5f}, Δ macro F1 {row['delta_macro_f1_vs_original_selected']:+.5f}; higher FP counts in {', '.join(up) or 'no labels'}; lower F1 in {', '.join(down) or 'no labels'}. Deltas refer to the original selected ConvNeXt; same-backbone deltas are also retained in the CSV. These are reused-validation findings, not independent performance claims.")
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    plot(histories,summary)
    print(json.dumps({'complete_matrix':complete,'selected_run':winner,'runs':[{k:r[k] for k in ('run','patient_macro_auroc','patient_macro_f1')} for r in summary]},indent=2))
    return result


def plot(histories,summary):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/runner/round2-v9/mpl-cache'));os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'artifacts/runner/round2-v9/cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,backbone in enumerate(('convnext_tiny','densenet121')):
        for entry in summary:
            if entry['backbone']!=backbone:continue
            h=histories[entry['run']];epochs=[r['epoch'] for r in h]
            axes[row,0].plot(epochs,[r['validation_patient_macro_auroc'] for r in h],marker='o',label=entry['run'])
            axes[row,1].plot(epochs,[r['validation_patient_weighted_bce'] for r in h],marker='o',label=entry['run'])
        axes[row,0].set_title(backbone+' — patient validation macro AUROC');axes[row,1].set_title(backbone+' — patient validation weighted BCE')
        for ax in axes[row]:ax.set_xlabel('Completed epoch');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'controlled-learning-curves.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--allow-incomplete',action='store_true');args=parser.parse_args();run(allow_incomplete=args.allow_incomplete)
