#!/usr/bin/env python3
"""Publish only aggregate Phase 6 outputs; patient records remain private."""
import json,shutil,sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trustretina.runner.verified_artifacts import read_stage
from trustretina.preprocessing.shared import PreprocessConfig
from trustretina.data.validation import file_hash
from full_experiment import write as atomic_write


def publish():
    base=ROOT/'artifacts/runner/phase6';registry=json.loads((base/'registry.json').read_text())
    if set(registry['stages'])!={'inputs','inference','analysis'} or any(s['status']!='complete' for s in registry['stages'].values()):raise ValueError('All three verified stages must complete')
    report,folder,v=read_stage(ROOT/registry['stages']['analysis']['bundle'])
    if report['status']!='complete' or report['protocol_sha256']!=registry['protocol_sha256'] or report['final_evaluation_access'] is not False:raise ValueError('Aggregate publication provenance mismatch')
    summary=json.loads((folder/'aggregate-summary.json').read_text());dest=ROOT/'results/reliability-v6';dest.mkdir(parents=True,exist_ok=True)
    # Explicit allowlist: never recursively copy the analysis bundle.
    for name in ('aggregate-summary.json','hamming_risk.png','any_error_risk.png'):shutil.copyfile(folder/name,dest/name)
    candidate=json.loads((folder/'candidate-pipeline.json').read_text())
    preprocessor=PreprocessConfig()
    if preprocessor.fingerprint()!=registry['selected_provenance']['preprocessing_sha256']:raise ValueError('Candidate preprocessing identity changed')
    contract={**candidate,'class_order':summary['class_order'],'preprocessing':asdict(preprocessor),'preprocessing_sha256':preprocessor.fingerprint(),'protocol_commit_before_calibration_labels':registry['protocol_commit'],'quality_fit_role':'train','quality_features':registry['config']['quality_features'],'quality_training_input_index_sha256':registry['previous_inputs_sha256'],'role_manifest_sha256':{k:v['manifest_sha256'] for k,v in registry['roles'].items()},'artifact_storage':{'analysis_kernel_version':registry['stages']['analysis']['kernel_version'],'candidate_file':'full/candidate-pipeline.json','candidate_sha256':summary['candidate_pipeline_sha256'],'temperature_file':'full/temperature.json','temperature_sha256':registry['stages']['analysis']['artifact_hashes']['temperature.json'],'quality_normalizer_file':'full/quality-normalizer.npz','quality_normalizer_sha256':registry['stages']['analysis']['artifact_hashes']['quality-normalizer.npz']},'selected_checkpoint_kernel_version':registry['selected_kernel_version'],'final_evaluation_status':'sealed during Phase6; future research evaluation requires its own authorized execution','clinical_automatic_acceptance_ready':False}
    (dest/'candidate-contract.json').write_text(json.dumps(contract,indent=2,sort_keys=True,allow_nan=False))
    shutil.copyfile(dest/'candidate-contract.json',base/'candidate-contract.json')
    registry['candidate_contract_sha256']=file_hash(dest/'candidate-contract.json');registry['candidate_pipeline_sha256']=summary['candidate_pipeline_sha256'];registry['temperature']=summary['temperature'];registry['research_candidate_frozen']=True
    atomic_write(base/'registry.json',registry)
    lines=['# Phase 6 conditional development results','',
           'Selected ConvNeXt-Tiny epoch two. Final evaluation remained sealed. No backbone retraining or acceptance-policy fit.','',
           f"Shared temperature **T={summary['temperature']:.6f}**, fitted only on calibration patients after mean eye probabilities. Calibration fit statistics are **in sample**, not generalization evidence.",'',
           '| Role | Metric | Raw | Calibrated |','|---|---|---:|---:|']
    for role in ('calibration','validation'):
        d=summary['diagnostics'][role]
        for metric in ('component_equal_nll','patient_nll','brier','macro_ece_10_equal_width_bins'):
            lines.append(f"| {role} | {metric} | {d['raw'][metric]:.6f} | {d['calibrated'][metric]:.6f} |")
    lines+=['','Validation already selected the backbone. These comparisons are conditional development descriptions. Lower AURC is better; no reliability strategy or cutoff is selected.','',
            '| Predictor / fixed strategy | Hamming AURC | Any-label-error AURC | Hamming AURC bootstrap 95% |','|---|---:|---:|---|']
    for name,item in summary['development_comparisons'].items():
        lo,hi=item['uncertainty']['hamming_aurc_percentile_95']
        lines.append(f"| {name} | {item['hamming_aurc']:.6f} | {item['any_error_aurc']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines+=['','Intervals use 200 resamples of whole containment components, conditional on the chosen backbone and fixed fitted temperature. They are not selection-adjusted, sparse-class or clinical confidence guarantees.','',
            '| Label | Validation positive / negative patients | Calibration positive / negative patients |','|---|---:|---:|']
    for label in summary['class_order']:
        val=summary['support']['validation'][label];cal=summary['support']['calibration'][label]
        lines.append(f"| {label} | {val['positive_patients']} / {val['negative_patients']} | {cal['positive_patients']} / {cal['negative_patients']} |")
    lines+=['',
            f"MC Dropout works at the already-trained p=0.2 head only: twenty seeded passes, both eyes aggregated within each pass. Stochastic depth and normalization remain in eval. It does not measure all backbone uncertainty. Quality uses training-only feature ECDFs and is an experimental technical signal, not clinical gradability.",'',
            f"Grad-CAM: {summary['gradcam_cases']} validation patients, {summary['gradcam_maps']} maps, {summary['gradcam_zero_maps']} zero maps. All predicted-positive labels are saved separately from the explanation target. Private overlays are canonical model-input explanations, not verified lesion locations.",'',
            '**Deployment default: REVIEW REQUIRED for every patient.** No independent supported policy partition/risk contract exists, so no automatic acceptance cutoff is fitted. All three raw/calibrated ablations are retained, including negative findings.','',
            '[Full permitted aggregates](aggregate-summary.json) · [Hamming curves](hamming_risk.png) · [Any-label-error curves](any_error_risk.png) · [Pinned run registry](run-registry.json)','',
            'Private artifacts contain the fitted parameter/provenance, candidate manifest, predictions, pass arrays, quality normalizer/signals, complete tied-threshold curves, fit diagnostics and Grad-CAM images. Use STATUS.md recovery commands. Final evaluation requires a separate authorized task with this frozen research candidate.','']
    (dest/'README.md').write_text('\n'.join(lines));print(json.dumps({'temperature':summary['temperature'],'candidate_pipeline_sha256':summary['candidate_pipeline_sha256']}))

if __name__=='__main__':publish()
