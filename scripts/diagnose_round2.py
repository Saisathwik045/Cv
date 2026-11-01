#!/usr/bin/env python3
"""Checksum-bound historical/development audit only. Never opens holdout prediction arrays."""
import json,sys,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trustretina.runner.verified_artifacts import read_stage
from trustretina.data.validation import file_hash
from trustretina.training.comparison import comparison_metrics,training_weights
from trustretina.config import CLASSES
import numpy as np


def run():
    saved=json.loads((ROOT/'artifacts/runner/full-v5/registry.json').read_text());reports={};digests={}
    for item in saved['runs']:
        report,folder,verified=read_stage(ROOT/item['attempts'][-1]['bundle'])
        if report['status']!='complete' or not report['strict_best_reload']:raise ValueError('Prior training incomplete')
        with np.load(folder/'validation-patients.npz',allow_pickle=False) as f:
            metrics=comparison_metrics(f['labels'],f['probabilities'])
        if metrics!=report['metrics']['patient']:raise ValueError('Historical validation metrics do not reproduce')
        history=json.loads((folder/'history.json').read_text());best_auc=max(history,key=lambda r:r['validation_patient_macro_auroc'])
        reports[item['backbone']]={'history':history,'selected_epoch':report['selection']['best_epoch']+1,'selected_validation_metrics':metrics,'recorded_max_auroc_epoch':best_auc['epoch'],'recorded_max_auroc':best_auc['validation_patient_macro_auroc'],'worker_seconds':report['elapsed_seconds'],'mean_epoch_seconds':statistics.mean(h['epoch_seconds'] for h in history),'extrapolated_25_epoch_worker_hours':report['elapsed_seconds']/len(history)*25/3600,'positive_weights':report['positive_weights']}
        digests[item['backbone']]={'report_sha256':verified['artifact_sha256'],'history_sha256':file_hash(folder/'history.json'),'best_sha256':report['output_hashes']['best.pt'],'kernel_version':verified['kernel_version']}
    if max(reports,key=lambda n:reports[n]['selected_validation_metrics']['macro_auroc'])!='convnext_tiny':raise ValueError('Historical winner differs')
    cache,folder,verification=read_stage(ROOT/'artifacts/runner/full-v5-inputs');index=json.loads((folder/'inputs.json').read_text());roles={};sets={}
    frozen=ROOT/'artifacts/runner/contained-release-v4/frozen-splits'
    if file_hash(frozen/'manifest.json')!=index['release_sha256']:raise ValueError('Frozen root manifest changed')
    for role in ('train','validation'):
        if file_hash(frozen/(role+'.json'))!=index['roles'][role]['parent_manifest_sha256']:raise ValueError('Frozen role changed')
    for role in ('train','validation'):
        item=index['roles'][role];rows=json.loads((folder/item['rows_file']).read_text());patients={}
        for r in rows:
            p=patients.setdefault(r['patient_id'],{'labels':r['labels'],'group':r['component_id'],'eyes':[]})
            if p['labels']!=r['labels'] or p['group']!=r['component_id']:raise ValueError('Within-patient mismatch')
            p['eyes'].append(r['eye'])
        if any(sorted(v['eyes'])!=['left','right'] for v in patients.values()):raise ValueError('Both eyes required')
        sets[role]={'patients':set(patients),'groups':{v['group'] for v in patients.values()}}
        groups={g:sum(v['group']==g for v in patients.values()) for g in sets[role]['groups']}
        support={}
        for j,c in enumerate(CLASSES):
            positives=[p for p in patients.values() if p['labels'][j]];negatives=[p for p in patients.values() if not p['labels'][j]]
            support[c]={'positive_patients':len(positives),'negative_patients':len(negatives),'positive_components':len({p['group'] for p in positives}),'negative_components':len({p['group'] for p in negatives})}
        roles[role]={'patients':len(patients),'images':len(rows),'components':len(groups),'largest_component':max(groups.values()),'support':support,'manifest_sha256':item['parent_manifest_sha256']}
    if any(sets['train'][k]&sets['validation'][k] for k in ('patients','groups')):raise ValueError('Role leakage')
    # Historical final results are verified for preservation only, in a separate
    # section never consumed by the queue/hyperparameter/threshold code.
    p7=json.loads((ROOT/'artifacts/runner/phase7/registry.json').read_text());v=json.loads((ROOT/p7['stages']['analysis']['bundle']/'verification.json').read_text());p=ROOT/p7['stages']['analysis']['bundle']/'download'/v['report_relative_path']
    if file_hash(p)!=v['artifact_sha256']:raise ValueError('Historical final report changed')
    final_report=json.loads(p.read_text());summary=p.parent/'aggregate-summary.json'
    if file_hash(summary)!=final_report['output_hashes']['aggregate-summary.json']:raise ValueError('Historical final aggregate changed')
    final=json.loads(summary.read_text())
    opaque={}
    for name in ('final-predictions.npz','intervals.json','configuration.json','temperature.json','quality-normalizer.npz'):
        if file_hash(p.parent/name)!=final_report['output_hashes'][name]:raise ValueError('Historical final artifact changed')
        opaque[name]=final_report['output_hashes'][name]
    historical={'preserved_opaque_artifact_hashes':opaque,'preservation_only_not_a_tuning_input':True,'macro_auroc':final['primary_estimate'],'macro_f1':final['estimates']['raw.patient_equal.classification.macro_f1'],'aggregate_sha256':file_hash(summary),'research_report_sha256':file_hash(ROOT/'results/final-v7/REPORT.md'),'selected_checkpoint_sha256':digests['convnext_tiny']['best_sha256'],'no_holdout_arrays_loaded':True}
    (ROOT/'artifacts/runner/round2-v9/historical-preservation.json').write_text(json.dumps(historical,indent=2))
    result={'schema':9,'source_runs':digests,'training':reports,'development_roles':roles,'existing_cache':{'kernel_version':verification['kernel_version'],'inputs_sha256':file_hash(folder/'inputs.json'),'report_sha256':verification['artifact_sha256']},'fresh_evaluation_required':True,'original_roles_unchanged':True,'historical_final_preservation_record':'private artifacts/runner/round2-v9/historical-preservation.json; not a tuning input'}
    (ROOT/'results/round2-v9/development-audit.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    print(json.dumps({'training':{n:{k:x[k] for k in ('selected_epoch','recorded_max_auroc_epoch','recorded_max_auroc','mean_epoch_seconds','extrapolated_25_epoch_worker_hours')} for n,x in reports.items()},'roles':roles,'historical_preservation_verified':True},indent=2))


if __name__=='__main__':run()
