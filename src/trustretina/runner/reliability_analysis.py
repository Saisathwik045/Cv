"""One CPU temperature fit and conditional development ablations; final evaluation sealed."""
import json,time,shutil,subprocess,sys,platform
from pathlib import Path
import numpy as np
from trustretina.config import CLASSES
from trustretina.data.validation import file_hash
from trustretina.evaluation.metrics import aggregate_patient_mean
from trustretina.reliability.development import (quality_midrank,patient_quality,fit_shared_temperature,temperature_probabilities,signals,coverage_curve,fixed_coverage_summary,calibration_diagnostics)
from trustretina.reliability.candidate import candidate_outputs
from .resumed_review import finish,progress


def json_write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))


def supplying_support(y,components):
    return {name:{'positive_patients':int(y[:,j].sum()),'negative_patients':int(len(y)-y[:,j].sum()),'positive_supplying_components':len(set(np.asarray(components)[y[:,j]==1])),'negative_supplying_components':len(set(np.asarray(components)[y[:,j]==0]))} for j,name in enumerate(CLASSES)}


def bootstrap_aurc(y,p,strategy_scores,components,*,replicates,seed):
    groups=sorted(set(components));mapping={g:np.flatnonzero(np.asarray(components)==g) for g in groups};rng=np.random.default_rng(seed);values={name:[] for name in strategy_scores}
    for _ in range(replicates):
        idx=np.concatenate([mapping[groups[j]] for j in rng.integers(0,len(groups),len(groups))])
        for name,score in strategy_scores.items():values[name].append(coverage_curve(y[idx],p[idx],score[idx])['hamming_aurc'])
    baseline=np.asarray(values['confidence'])
    return {name:{'hamming_aurc_percentile_95':np.quantile(v,[.025,.975]).tolist(),'paired_hamming_aurc_difference_vs_confidence_percentile_95':np.quantile(np.asarray(v)-baseline,[.025,.975]).tolist(),'replicates':replicates,'seed':seed,'unit':'whole containment component; conditional descriptive intervals, not selection-adjusted or clinical bounds'} for name,v in values.items()}


def run(spec,*,input_root,output,nonce,source_commit):
    start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False);cfg=spec['config']
    report={'nonce':nonce,'source_commit':source_commit,'training':False,'gpu_requested':False,'stage':'phase6_analysis','status':'incomplete','protocol_sha256':spec['protocol_sha256'],'release_sha256':spec['release_sha256'],'checkpoint_sha256':spec['checkpoint_sha256'],'final_evaluation_access':False,'risk_policy_fitted':False}
    try:
        if cfg['final_evaluation_access'] or cfg['risk_policy_fitting'] or cfg['backbone_retraining']:raise ValueError('Phase6 analysis scope violation')
        index_paths=[p for p in Path(input_root).rglob('inputs.json') if file_hash(p)==spec['inputs_sha256']]
        gpu_paths=[p for p in Path(input_root).rglob('stage-report.json') if file_hash(p)==spec['inference_report_sha256']]
        if len(index_paths)!=1 or len(gpu_paths)!=1:raise ValueError('Verified analysis sources missing/ambiguous')
        root=index_paths[0].parent;index=json.loads(index_paths[0].read_text());gpu=gpu_paths[0].parent;inference=json.loads(gpu_paths[0].read_text())
        if inference['status']!='complete' or inference['protocol_sha256']!=spec['protocol_sha256'] or inference['checkpoint_sha256']!=cfg['checkpoint_sha256'] or not inference['model_state_unchanged'] or index['protocol_sha256']!=spec['protocol_sha256'] or set(index['roles'])!={'validation','calibration'}:raise ValueError('Inference/protocol invalid')
        for name,digest in inference['output_hashes'].items():
            p=(gpu/name).resolve()
            if not p.is_relative_to(gpu.resolve()) or p.is_symlink() or file_hash(p)!=digest:raise ValueError('Inference artifact changed')
        if file_hash(root/index['quality_file'])!=index['quality_sha256']:raise ValueError('Training quality features changed')
        with np.load(root/index['quality_file'],allow_pickle=False) as q:features={k:q[k] for k in q.files}
        data={};rows={};qualities={};supports={}
        for role,item in index['roles'].items():
            if file_hash(root/item['rows_file'])!=item['rows_sha256']:raise ValueError('Frozen role row index changed')
            rows[role]=json.loads((root/item['rows_file']).read_text())
            with np.load(gpu/(role+'.npz'),allow_pickle=False) as f:data[role]={k:f[k] for k in f.files}
            d=data[role];r=rows[role]
            if d['image_ids'].tolist()!=[x['path'] for x in r] or d['image_patient_ids'].tolist()!=[x['patient_id'] for x in r] or d['image_eyes'].tolist()!=[x['eye'] for x in r]:raise ValueError('Image alignment changed')
            aggregated=aggregate_patient_mean([x['patient_id'] for x in r],[x['eye'] for x in r],np.asarray([x['labels'] for x in r]),d['image_probabilities'],[x['component_id'] for x in r])
            if d['patient_ids'].tolist()!=aggregated['patient_ids'] or not np.array_equal(d['labels'],aggregated['labels']) or not np.array_equal(d['probabilities'],aggregated['probabilities']) or d['component_ids'].tolist()!=aggregated['component_ids']:raise ValueError('Patient aggregation changed')
            if d['mc_patient_probabilities'].shape!=(cfg['mc_passes'],len(d['labels']),8) or d['mc_seeds'].tolist()!=list(range(cfg['mc_'+role+'_seed'],cfg['mc_'+role+'_seed']+cfg['mc_passes'])):raise ValueError('MC seed/pass contract changed')
            eye_quality=quality_midrank(features['train'],features[role]);qualities[role]=patient_quality(r,eye_quality,d['patient_ids'].tolist());supports[role]=supplying_support(d['labels'],d['component_ids'])
        if set(data['validation']['patient_ids'])&set(data['calibration']['patient_ids']) or set(data['validation']['component_ids'])&set(data['calibration']['component_ids']):raise ValueError('Development role leakage')
        # The one and only fitted temperature uses calibration targets exclusively.
        d=data['calibration'];fit=fit_shared_temperature(d['probabilities'],d['labels'],d['component_ids'].tolist(),role='calibration',allow_fitting=True)
        fit.update(config_sha256=spec['config_sha256'],protocol_sha256=spec['protocol_sha256'],checkpoint_sha256=spec['checkpoint_sha256'],calibration_manifest_sha256=index['roles']['calibration']['parent_manifest_sha256'],calibration_patients=len(d['labels']),calibration_components=len(set(d['component_ids'])),inference_report_sha256=spec['inference_report_sha256'],source_commit=source_commit)
        json_write(output/'temperature.json',fit);temperature=fit['temperature'];diagnostics={};role_signals={};candidate={}
        for role,d in data.items():
            raw=d['probabilities'];calibrated=temperature_probabilities(raw,temperature);mc_raw=d['mc_patient_probabilities'];mc_cal=np.stack([temperature_probabilities(p,temperature) for p in mc_raw])
            if not np.array_equal(raw>=.5,calibrated>=.5):raise ValueError('Shared temperature unexpectedly changed binary decisions')
            role_signals[role]={'raw':signals(raw,mc_raw,qualities[role]),'calibrated':signals(calibrated,mc_cal,qualities[role])}
            diagnostics[role]={'role_interpretation':'in-sample calibration fit diagnostic' if role=='calibration' else 'conditional development diagnostic after model selection, not held-out final evidence','raw':calibration_diagnostics(d['labels'],raw,d['component_ids']),'calibrated':calibration_diagnostics(d['labels'],calibrated,d['component_ids'])}
            candidate[role]=candidate_outputs(raw,mc_raw,qualities[role],temperature=temperature)
            np.savez_compressed(output/(role+'-candidate.npz'),**d,calibrated_probabilities=calibrated,calibrated_mc_patient_probabilities=mc_cal,quality=qualities[role],raw_uncertainty=role_signals[role]['raw']['uncertainty'],calibrated_uncertainty=role_signals[role]['calibrated']['uncertainty'])
            json_write(output/(role+'-decisions.json'),[{'patient_id':pid,'positive_labels':labels,'decision':decision,'explanation_target_index':int(target)} for pid,labels,decision,target in zip(d['patient_ids'],candidate[role]['positive_labels'],candidate[role]['decision'],candidate[role]['explanation_target_indices'])])
        np.savez_compressed(output/'quality-normalizer.npz',sorted_training_feature_distributions=np.sort(features['train'],axis=0));json_write(output/'fit-diagnostics.json',diagnostics)
        d=data['validation'];comparisons={};curves={}
        for mode in ('raw','calibrated'):
            p=d['probabilities'] if mode=='raw' else temperature_probabilities(d['probabilities'],temperature)
            scores={k:v for k,v in role_signals['validation'][mode].items() if k not in ('uncertainty','quality')}
            intervals=bootstrap_aurc(d['labels'],p,scores,d['component_ids'],replicates=cfg['bootstrap_replicates'],seed=cfg['bootstrap_seed'])
            for name,score in scores.items():
                if time.monotonic()-start>cfg['maximum_cpu_analysis_seconds']-60:raise TimeoutError('CPU analysis cap')
                key=mode+'__'+name;curve=coverage_curve(d['labels'],p,score);curves[key]=curve
                comparisons[key]={'hamming_aurc':curve['hamming_aurc'],'any_error_aurc':curve['any_error_aurc'],'fixed_coverage':fixed_coverage_summary(d['labels'],p,score,cfg['coverage_grid']),'uncertainty':intervals[name],'selection':'fixed prespecified ablation; no strategy or deployment threshold selected'}
                np.savez_compressed(output/(key+'-curve.npz'),**curve)
        json_write(output/'development-comparisons.json',comparisons)
        cams=json.loads((gpu/'gradcam-records.json').read_text())
        for record in cams:
            record['target_label']=CLASSES[record['target_index']];record['positive_labels']=[CLASSES[i] for i in record['positive_indices']]
        json_write(output/'gradcam-records.json',cams)
        for p in gpu.glob('gradcam-*'):
            if p.suffix in ('.npz','.png'):shutil.copyfile(p,output/p.name)
        pipeline={'schema':6,'checkpoint_sha256':cfg['checkpoint_sha256'],'source_commit':source_commit,'config_sha256':spec['config_sha256'],'protocol_sha256':spec['protocol_sha256'],'release_sha256':cfg['release_sha256'],'selected_backbone':'convnext_tiny','best_epoch':2,'deterministic_predictor':'mean left/right sigmoid probabilities','temperature':temperature,'calibration_order':cfg['calibration_order'],'threshold':.5,'mc_passes':cfg['mc_passes'],'mc_validation_seed':cfg['mc_validation_seed'],'mc_calibration_seed':cfg['mc_calibration_seed'],'mc_scope':'trained head.0 Dropout p=0.2; stochastic depth and normalization eval; no inserted layers','quality_normalizer_sha256':file_hash(output/'quality-normalizer.npz'),'temperature_sha256':file_hash(output/'temperature.json'),'deployment_default':'REVIEW REQUIRED','automatic_acceptance_policy_fitted':False,'reliability_strategy_selection':'none; all three fixed exploratory ablations retained','gradcam_layer':cfg['gradcam_layer'],'gradcam_target':cfg['gradcam_target'],'final_evaluation_accessed':False,'readiness':'frozen research candidate; needs separately authorized held-out evaluation; automatic acceptance unsupported'}
        json_write(output/'candidate-pipeline.json',pipeline)
        import matplotlib;matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        for risk in ('hamming_risk','any_error_risk'):
            fig,ax=plt.subplots(figsize=(9,5))
            for name,curve in curves.items():ax.plot(curve['coverage'],curve[risk],label=name)
            ax.set(xlabel='Validation patient coverage (exploratory)',ylabel=risk.replace('_',' '),title='Conditional development curves; no acceptance policy fitted',ylim=(0,1));ax.legend(fontsize=7);fig.tight_layout();fig.savefig(output/(risk+'.png'),dpi=160);plt.close(fig)
        fig,axes=plt.subplots(2,4,figsize=(13,7))
        for j,ax in enumerate(axes.flat):
            for mode in ('raw','calibrated'):
                bins=diagnostics['calibration'][mode]['per_class'][j]['bins'];points=[b for b in bins if b['n']]
                ax.plot([b['mean_probability'] for b in points],[b['positive_fraction'] for b in points],marker='o',label=mode)
            ax.plot([0,1],[0,1],'k--',linewidth=.7);ax.set(title=CLASSES[j],xlim=(0,1),ylim=(0,1));ax.legend(fontsize=6)
        fig.suptitle('Calibration fit diagnostics IN SAMPLE; rare classes have sparse bins');fig.tight_layout();fig.savefig(output/'calibration-in-sample.png',dpi=160);plt.close(fig)
        # Permitted aggregate subset excludes IDs, sample arrays and attainable thresholds.
        public_diagnostics={role:{'interpretation':v['role_interpretation'],**{mode:{k:value for k,value in v[mode].items() if k!='per_class'} for mode in ('raw','calibrated')}} for role,v in diagnostics.items()}
        public_comparisons={k:{'hamming_aurc':v['hamming_aurc'],'any_error_aurc':v['any_error_aurc'],'uncertainty':v['uncertainty'],'fixed_coverage':[{'target_coverage':row['target_coverage'],'coverage':row['coverage'],'accepted':row['accepted'],'hamming_risk':row['hamming_risk'],'any_error_risk':row['any_error_risk']} for row in v['fixed_coverage']]} for k,v in comparisons.items()}
        summary={'class_order':CLASSES,'temperature':temperature,'fit':{k:v for k,v in fit.items() if k not in ('source_commit',)},'diagnostics':public_diagnostics,'support':supports,'development_comparisons':public_comparisons,'mc_supported':True,'mc_scope':pipeline['mc_scope'],'gradcam_cases':len(cams),'gradcam_maps':2*len(cams),'gradcam_zero_maps':inference['gradcam_zero_maps'],'all_deterministic_binary_decisions_unchanged':True,'deployment_default':'REVIEW REQUIRED','automatic_acceptance_policy_fitted':False,'candidate_pipeline_sha256':file_hash(output/'candidate-pipeline.json'),'final_evaluation_accessed':False,'limitations':['Calibration diagnostics are in sample','Validation selected the backbone; ablations are conditional development evidence','Head dropout omits backbone epistemic uncertainty','Heuristic quality is not clinical gradability','Sparse classes and duplicate-connected dependence limit claims','No automatic acceptance cutoff or clinical guarantee supported']}
        json_write(output/'aggregate-summary.json',summary)
        (output/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        report.update(status='complete',temperature=temperature,calibration_fitted=True,fit_objective=fit['objective'],mc_supported=True,all_deterministic_binary_decisions_unchanged=True,candidate_pipeline_sha256=file_hash(output/'candidate-pipeline.json'),deployment_default='REVIEW REQUIRED',gradcam_cases=len(cams),gradcam_zero_maps=inference['gradcam_zero_maps'],packages={'python':platform.python_version(),'numpy':np.__version__,'scipy':__import__('scipy').__version__,'sklearn':__import__('sklearn').__version__})
    except Exception as e:
        import traceback;traceback.print_exc();report.update(error_type=type(e).__name__,error=str(e))
    report['elapsed_seconds']=time.monotonic()-start;finish(output,report);progress('phase6_analysis_finished',status=report['status']);return report
