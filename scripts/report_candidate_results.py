#!/usr/bin/env python3
"""Report only verified complete v14 validation outputs; no calibration reader or inference."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from trustretina.data.validation import file_hash
from trustretina.training.comparison import comparison_metrics
import candidate_experiment as controller


def summarize_validation(labels,probabilities,component_ids=None):
    y=np.asarray(labels);p=np.asarray(probabilities)
    m=comparison_metrics(y,p);conf=multilabel_confusion_matrix(y,p>=.5)
    rows=[]
    for j,c in enumerate('NDGCAHMO'):
        tn,fp,fn,tp=map(int,conf[j].ravel());r=m['per_class'][c]
        groups=np.asarray(component_ids) if component_ids is not None else None
        if groups is not None and groups.shape!=(len(y),):raise ValueError('One containment identity per patient required')
        rows.append({'class':c,'positive':int(r['positive']),'negative':int(r['negative']),'AUROC':r['auroc'],'AP':r['average_precision'],'precision':r['precision'],'recall':r['recall'],'F1':r['f1'],'TP':tp,'FP':fp,'FN':fn,'TN':tn,'predicted_positive':tp+fp,'FPR':fp/(fp+tn) if fp+tn else None,'FNR':fn/(tp+fn) if tp+fn else None,'precision_defined':bool(tp+fp),'recall_defined':bool(tp+fn),'positive_components':len(set(groups[y[:,j]==1])) if groups is not None else None,'negative_components':len(set(groups[y[:,j]==0])) if groups is not None else None})
    return m,rows


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'results/candidate-v14');args=parser.parse_args()
    p=json.loads((controller.BASE/'registry.json').read_text());controller.verify_plan(p)
    if not all(r['status']=='complete' for r in p['runs']):raise ValueError('Both exact fits must be verified complete; do not report an incomplete matrix as a winner')
    aggregate=[];per_class=[];histories=[];proof=[];aligned=None
    for run in p['runs']:
        attempt=run['attempts'][-1];bundle=Path(attempt['artifact_bundle']);reportpath=bundle/'download/full/stage-report.json';folder=reportpath.parent;report=json.loads(reportpath.read_text())
        if file_hash(reportpath)!=attempt['report_sha256'] or report['plan_sha256']!=p['plan_sha256'] or report['run_id']!=run['id']:raise ValueError('Report/source identity changed')
        if report['status']!='complete' or not all(report.get(k) for k in ('strict_best_reload','model_state_invariance','calibration_inference_after_freeze')):raise ValueError('Incomplete strict-freeze evidence')
        for flag in ('final_evaluation_access','calibration_fitted','threshold_fitted','risk_policy_fitted','calibration_labels_accessed'):
            if report.get(flag) is not False:raise ValueError('Forbidden role access or parameter fitting in report')
        if not np.allclose(report['positive_weights'],p['positive_weights'],rtol=0,atol=1e-12):raise ValueError('TRAIN-only loss weights differ')
        for name,digest in report['output_hashes'].items():
            target=(folder/name).resolve()
            if not target.is_relative_to(folder.resolve()) or file_hash(target)!=digest:raise ValueError('Private artifact hash changed')
        d=np.load(folder/'validation-patients.npz',allow_pickle=False);eyes=np.load(folder/'validation-images.npz',allow_pickle=False);passes=np.load(folder/'validation-passes.npz',allow_pickle=False)
        y=d['labels'];scores=d['probabilities']
        if scores.shape!=(525,8) or y.shape!=scores.shape or len(set(d['patient_ids']))!=525:raise ValueError('Invalid validation alignment')
        if passes['probabilities'].shape!=(20,525,8) or not np.array_equal(passes['patient_ids'],d['patient_ids']) or not np.array_equal(passes['seeds'],np.arange(62026,62046)):raise ValueError('Stochastic pass/seed alignment changed')
        for pid,prob,label in zip(d['patient_ids'],scores,y):
            mask=eyes['patient_ids']==pid
            if mask.sum()!=2 or set(eyes['eyes'][mask])!={'left','right'} or not np.array_equal(eyes['labels'][mask],np.tile(label,(2,1))) or not np.allclose(eyes['probabilities'][mask].astype(float).mean(0),prob,atol=1e-12,rtol=0):raise ValueError('Mean-eye target/prediction mismatch')
        if aligned is None:aligned=(d['patient_ids'].copy(),d['component_ids'].copy(),y.copy())
        if any(not np.array_equal(a,b) for a,b in zip(aligned,(d['patient_ids'],d['component_ids'],y))):raise ValueError('Matched fit sample alignment changed')
        m,rows=summarize_validation(y,scores,d['component_ids'])
        for key in ('macro_auroc','macro_auprc_average_precision','macro_f1','micro_f1','subset_exact_match_accuracy','labelwise_binary_accuracy','hamming_loss'):
            if abs(m[key]-report['metrics'][key])>1e-12:raise ValueError('Recomputed metric differs')
        record={'run_id':run['id'],'resolution':run['image_size'],'epochs':report['completed_epochs'],'best_epoch':report['selection']['best_epoch']+1,'macro_AUROC':m['macro_auroc'],'macro_AP':m['macro_auprc_average_precision'],'macro_F1':m['macro_f1'],'micro_F1':m['micro_f1'],'exact_match_accuracy':m['subset_exact_match_accuracy'],'labelwise_accuracy':m['labelwise_binary_accuracy'],'Hamming_loss':m['hamming_loss'],'patients':525,'threshold':.5,'worker_seconds':report['elapsed_seconds']}
        aggregate.append(record);per_class.extend({'run_id':run['id'],**r} for r in rows)
        for h in report['history']:histories.append({'run_id':run['id'],**{k:v for k,v in h.items() if k!='validation_patient_metrics'}})
        proof.append({'run_id':run['id'],'kernel_version':attempt['kernel_version'],'source_commit':report['source_commit'],'plan_sha256':report['plan_sha256'],'report_sha256':attempt['report_sha256'],'selected_checkpoint_sha256':report['selected_checkpoint_sha256'],'all_artifact_hashes':report['output_hashes'],'private_execution_log_sha256':file_hash(bundle/'execution.log'),'strict_best_reload':True,'model_state_invariance':True,'post_freeze_calibration_inference':True,'calibration_labels_accessed':False})
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'validation-comparison.csv',aggregate);write_csv(out/'validation-per-class.csv',per_class);write_csv(out/'training-history.csv',histories)
    controller.write(out/'execution-results.json',{'scope':'Reused-validation development; not independent confirmation','plan_sha256':p['plan_sha256'],'comparison':aggregate,'per_class':per_class,'artifact_provenance':proof,'quota_baseline':p['quota_baseline'],'sessions_submitted':sum(len(r['attempts']) for r in p['runs']),'automatic_promotion':False,'temperature_or_threshold_fit':False,'historical_final_access':False,'demo_changed':False})
    lines=['# V14 two-fit development results','','Both frozen fits completed and outputs passed source/nonce/report/checksum verification. These are reused-validation development results, not independent confirmation. No model promotion, temperature/threshold fitting, historical final access or demo modification.','','| Fit | Epochs / best | Macro AUROC | Macro AP | Macro F1 | Micro F1 |','|---|---:|---:|---:|---:|---:|']
    for r in aggregate:lines.append(f"| {r['resolution']} | {r['epochs']} / {r['best_epoch']} | {r['macro_AUROC']:.4f} | {r['macro_AP']:.4f} | {r['macro_F1']:.4f} | {r['micro_F1']:.4f} |")
    lines+=['','Raw patient probabilities are float64 mean left/right sigmoid probabilities; fixed p≥0.5. TRAIN-only ratio BCE; validation macro AUROC selects each checkpoint.','','| Class | Resolution | Positive / negative | Precision | Recall | F1 | TP | FP | FN |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in 'NDGCAHMO':
        for run in p['runs']:
            r=next(r for r in per_class if r['run_id']==run['id'] and r['class']==c);lines.append(f"| {c} | {run['image_size']} | {r['positive']} / {r['negative']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['F1']:.4f} | {r['TP']} | {r['FP']} | {r['FN']} |")
    lines+=['','## Explicit class-specific limitations','']
    for c in 'HO':
        vals=[next(r for r in per_class if r['run_id']==run['id'] and r['class']==c) for run in p['runs']];lines.append(f"- {c} false positives: 320={vals[0]['FP']}, 224={vals[1]['FP']}; false negatives: 320={vals[0]['FN']}, 224={vals[1]['FN']}.")
    for c in 'DAM':
        vals=[next(r for r in per_class if r['run_id']==run['id'] and r['class']==c) for run in p['runs']];lines.append(f"- {c} missed positives: 320={vals[0]['FN']}, 224={vals[1]['FN']}, among {vals[0]['positive']} validation positives.")
    by_class={(r['run_id'],r['class']):r for r in per_class}
    a,b=aggregate
    lines+=['',f"The descriptive320−224 differences are macro AUROC{a['macro_AUROC']-b['macro_AUROC']:+.5f}, macro AP{a['macro_AP']-b['macro_AP']:+.5f}, macro F1{a['macro_F1']-b['macro_F1']:+.5f} and micro F1{a['micro_F1']-b['micro_F1']:+.5f}. No independent confirmation or significance claim follows.",'','Additional regressions must not be hidden:']
    for c in ('C','G'):
        x,z=[by_class[(run['id'],c)] for run in p['runs']]
        lines.append(f"- {c}:320 precision/recall/F1={x['precision']:.4f}/{x['recall']:.4f}/{x['F1']:.4f}, FP/FN={x['FP']}/{x['FN']};224={z['precision']:.4f}/{z['recall']:.4f}/{z['F1']:.4f}, FP/FN={z['FP']}/{z['FN']}.")
    lines+=['','| Resolution | Exact-match accuracy | Labelwise binary accuracy | Hamming loss |','|---|---:|---:|---:|']
    for x in aggregate:lines.append(f"| {x['resolution']} | {x['exact_match_accuracy']:.4f} | {x['labelwise_accuracy']:.4f} | {x['Hamming_loss']:.4f} |")
    lines+=['','320 has lower exact-match accuracy despite its higher aggregate ranking/F1 scores. H detection remains weak (only4/16 positive patients detected at320), and C/A/G/H tradeoffs preclude a claim of uniform class improvement. Labelwise accuracy is not diagnostic accuracy. No independent-cohort endpoint or grouped interval is reported here; those remain unexecuted, separately gated research steps.']
    lines+=['','All-eight AUROC/AP and TN are in validation-per-class.csv. Sparse H support and reused validation limit interpretation. BRSET remains an independent access/compatibility blocker. The original demo remains REVIEW REQUIRED. Large checkpoints, aligned predictions, stochastic arrays and logs are retained in the pinned private job versions; hashes are recorded in execution-results.json.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
