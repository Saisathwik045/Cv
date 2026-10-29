"""Frozen final research analysis, paired uncertainty and all prespecified ablations. No fit path."""
import json,time,shutil,platform,subprocess,sys
from pathlib import Path
import numpy as np
from trustretina.config import CLASSES
from trustretina.data.validation import file_hash
from trustretina.evaluation.metrics import aggregate_patient_mean
from trustretina.evaluation.final_research import statistics,bootstrap
from trustretina.reliability.development import temperature_probabilities,patient_quality,signals,component_weights
from .resumed_review import finish,progress


def save_json(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))


def verify_report(path,digest):
    if file_hash(path)!=digest:raise ValueError('Pinned report bytes changed')
    report=json.loads(path.read_text())
    for name,expected in report['output_hashes'].items():
        p=(path.parent/name).resolve()
        if not p.is_relative_to(path.parent.resolve()) or p.is_symlink() or file_hash(p)!=expected:raise ValueError('Pinned source artifact changed')
    if report['status']!='complete':raise ValueError('Incomplete source stage')
    return report


def frozen_quality(normalizer,values):
    training=np.asarray(normalizer,dtype=float);v=np.asarray(values,dtype=float)
    if training.ndim!=2 or training.shape[1]!=3 or v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(training).all() or not np.isfinite(v).all() or not len(training) or (np.diff(training,axis=0)<0).any():raise ValueError('Frozen sorted normalizer/features invalid')
    return np.column_stack([(np.searchsorted(training[:,j],v[:,j],side='left')+np.searchsorted(training[:,j],v[:,j],side='right'))/(2*len(training)) for j in range(3)]).mean(1)


def run(spec,*,input_root,output,nonce,source_commit):
    start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False);cfg=spec['config'];contract=spec['contract']
    report={'stage':'phase7_analysis','nonce':nonce,'source_commit':source_commit,'training':False,'gpu_requested':False,'status':'incomplete','protocol_sha256':spec['protocol_sha256'],'candidate_contract_sha256':spec['candidate_contract_sha256'],'parameter_fitting':False,'final_evaluation_access':True,'backbone_training':False,'risk_policy_fitted':False}
    try:
        if not cfg['final_evaluation_authorized'] or cfg['role']!='final_evaluation' or cfg['parameter_fitting'] or cfg['backbone_training'] or cfg['risk_policy_fitting']:raise ValueError('Final analysis scope violation')
        index_paths=[p for p in Path(input_root).rglob('inputs.json') if file_hash(p)==spec['inputs_sha256']]
        inference_paths=[p for p in Path(input_root).rglob('stage-report.json') if file_hash(p)==spec['inference_report_sha256']]
        old_paths=[p for p in Path(input_root).rglob('stage-report.json') if file_hash(p)==spec['phase6_analysis_report_sha256']]
        if len(index_paths)!=1 or len(inference_paths)!=1 or len(old_paths)!=1:raise ValueError('Verified final/development sources missing/ambiguous')
        root=index_paths[0].parent;index=json.loads(index_paths[0].read_text());gpu=inference_paths[0].parent;old=old_paths[0].parent
        inference=verify_report(inference_paths[0],spec['inference_report_sha256']);prior=verify_report(old_paths[0],spec['phase6_analysis_report_sha256'])
        if index['role']!='final_evaluation' or index['manifest_sha256']!=spec['final_manifest_sha256'] or inference['candidate_contract_sha256']!=spec['candidate_contract_sha256'] or not inference['model_state_unchanged'] or inference['parameter_fitting'] is not False:raise ValueError('Final role/model scope mismatch')
        for key in ('rows','quality'):
            if file_hash(root/index[key+'_file'])!=index[key+'_sha256']:raise ValueError('Frozen input metadata/features changed')
        for filename,expected in [('temperature.json',contract['temperature_sha256']),('quality-normalizer.npz',contract['quality_normalizer_sha256']),('development-comparisons.json',spec['development_comparisons_sha256']),('candidate-pipeline.json',contract['artifact_storage']['candidate_sha256'])]:
            if file_hash(old/filename)!=expected:raise ValueError('Frozen parameter/normalizer/development thresholds changed')
        temperature=json.loads((old/'temperature.json').read_text())['temperature']
        if temperature!=contract['temperature']:raise ValueError('Frozen temperature differs')
        rows=json.loads((root/index['rows_file']).read_text())
        with np.load(gpu/'deterministic.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
        with np.load(gpu/'stochastic.npz',allow_pickle=False) as f:mc={k:f[k] for k in f.files}
        for key,expected in [('image_ids',[r['path'] for r in rows]),('image_patient_ids',[r['patient_id'] for r in rows]),('image_eyes',[r['eye'] for r in rows])]:
            if d[key].tolist()!=expected:raise ValueError('Final image alignment changed')
        y_image=np.asarray([r['labels'] for r in rows]);patients=aggregate_patient_mean([r['patient_id'] for r in rows],[r['eye'] for r in rows],y_image,d['image_probabilities'],[r['component_id'] for r in rows])
        if not np.array_equal(d['image_labels'],y_image) or patients['patient_ids']!=d['patient_ids'].tolist() or not np.array_equal(patients['labels'],d['labels']) or not np.array_equal(patients['probabilities'],d['probabilities']) or patients['component_ids']!=d['component_ids'].tolist() or len(d['labels'])!=cfg['expected_patients']:raise ValueError('Final patient/label/component aggregation mismatch')
        if mc['patient_ids'].tolist()!=d['patient_ids'].tolist() or mc['probabilities'].shape!=(20,len(d['labels']),8) or mc['seeds'].tolist()!=list(range(contract['mc_validation_seed'],contract['mc_validation_seed']+20)):raise ValueError('Frozen stochastic alignment/seeds changed')
        with np.load(old/'quality-normalizer.npz',allow_pickle=False) as f:normalizer=f['sorted_training_feature_distributions']
        with np.load(root/index['quality_file'],allow_pickle=False) as f:features=f['features']
        quality=patient_quality(rows,frozen_quality(normalizer,features),d['patient_ids'].tolist())
        raw=d['probabilities'];cal=temperature_probabilities(raw,temperature);mc_raw=mc['probabilities'];mc_cal=np.stack([temperature_probabilities(p,temperature) for p in mc_raw])
        if not np.array_equal(raw>=.5,cal>=.5):raise ValueError('Frozen positive temperature changed threshold decisions')
        predictions={'raw':raw,'calibrated':cal};scores={}
        for mode,p,samples in [('raw',raw,mc_raw),('calibrated',cal,mc_cal)]:
            sig=signals(p,samples,quality)
            scores.update({mode+'__'+k:v for k,v in sig.items() if k not in ('uncertainty','quality')})
        development=json.loads((old/'development-comparisons.json').read_text());cutoffs={k:v['fixed_coverage'] for k,v in development.items()}
        if set(cutoffs)!=set(scores) or any([v['target_coverage'] for v in values]!=cfg['coverage_grid'] for values in cutoffs.values()):raise ValueError('Frozen coverage strategy/grid mismatch')
        weights={'patient_equal':np.ones(len(raw)),'component_equal':component_weights(d['component_ids'])}
        estimates,result,curves=statistics(d['labels'],predictions,scores,weights,cutoffs,detailed=True)
        save_json(output/'point-estimates.json',estimates);save_json(output/'full-results.json',result)
        for name,curve in curves.items():np.savez_compressed(output/(name+'-curve.npz'),**curve)
        np.savez_compressed(output/'final-predictions.npz',**d,calibrated_probabilities=cal,mc_patient_probabilities=mc_raw,calibrated_mc_patient_probabilities=mc_cal,mc_seeds=mc['seeds'],quality=quality,raw_mc_variance=mc_raw.var(0,ddof=0),calibrated_mc_variance=mc_cal.var(0,ddof=0),**{'score__'+k:v for k,v in scores.items()})
        save_json(output/'deployment-decisions.json',[{'patient_id':pid,'positive_labels':[label for label,v in zip(CLASSES,row) if v>=.5],'decision':'REVIEW REQUIRED'} for pid,row in zip(d['patient_ids'],cal)])
        def checkpoint(done,total):
            progress('final_component_bootstrap',completed=done,total=total,elapsed_seconds=round(time.monotonic()-start))
            if time.monotonic()-start>cfg['maximum_cpu_analysis_seconds']-120:raise TimeoutError('Final CPU analysis cap')
        intervals,samples=bootstrap(d['labels'],predictions,scores,d['component_ids'],cutoffs,replicates=cfg['bootstrap_replicates'],seed=cfg['bootstrap_seed'],minimum_defined=cfg['minimum_defined_bootstrap_replicates'],progress_callback=checkpoint)
        save_json(output/'intervals.json',intervals);np.savez_compressed(output/'bootstrap-statistics.npz',**samples)
        supports={c:{'positive_patients':int(d['labels'][:,j].sum()),'negative_patients':int(len(raw)-d['labels'][:,j].sum()),'positive_supplying_components':len(set(d['component_ids'][d['labels'][:,j]==1])),'negative_supplying_components':len(set(d['component_ids'][d['labels'][:,j]==0]))} for j,c in enumerate(CLASSES)}
        # Aggregate publication excludes samples, all attainable thresholds and bin-level data.
        public=json.loads(json.dumps(result))
        for mode in public['metrics'].values():
            for value in mode.values():
                for per in value['calibration']['per_class'].values():per.pop('bins')
        for strategy in public['reliability'].values():
            for value in strategy.values():
                for grid in value['saved_validation_grid']:grid.pop('frozen_validation_threshold')
        primary_key='raw.patient_equal.classification.macro_auroc'
        summary={'schema':7,'primary_endpoint':cfg['primary_endpoint'],'primary_estimate':estimates[primary_key],'primary_interval':intervals[primary_key],'patients':len(raw),'images':len(rows),'components':len(set(d['component_ids'])),'class_order':CLASSES,'support':supports,'results':public,'estimates':estimates,'intervals':intervals,'temperature_reused':temperature,'candidate_contract_sha256':spec['candidate_contract_sha256'],'final_manifest_sha256':spec['final_manifest_sha256'],'all_binary_decisions_unchanged':True,'all_frozen_parameters_reused':True,'parameter_fitting':False,'policy_fitting':False,'deployment_default':'REVIEW REQUIRED','deployment_accept_count':0,'deployment_review_count':len(raw),'coverage_interpretation':'Saved validation cutoffs applied unchanged; actual final coverage differs. Full curves/AURC retrospective research, not an automatic acceptance policy.','interval_interpretation':'Conditional on frozen candidate; paired whole-containment-component percentile intervals, not simultaneous clinical guarantees','limitations':['One internal ODIR holdout, no external clinical validation','Patient-derived source labels do not establish eye-specific diagnosis','Rare classes and sparse retained subsets have limited precision','Head-only Dropout omits backbone/model uncertainty','Heuristic quality is not clinical gradability','Bounded duplicate assay cannot exclude every shared identity','Bootstrap does not propagate training/temperature-fit uncertainty','No clinical acceptable-risk contract or automatic acceptance cutoff']}
        save_json(output/'aggregate-summary.json',summary)
        save_json(output/'configuration.json',spec)
        for filename in ('temperature.json','quality-normalizer.npz','candidate-pipeline.json'):shutil.copyfile(old/filename,output/filename)
        import matplotlib;matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        for weighting in weights:
            for risk in ('hamming_risk','any_error_risk'):
                fig,ax=plt.subplots(figsize=(9,5))
                for name in scores:
                    curve=curves[name+'__'+weighting];ax.plot(curve['coverage'],curve[risk],label=name)
                ax.set(xlabel=weighting+' retrospective coverage',ylabel=risk.replace('_',' '),title='Frozen final cohort; research curves, no acceptance policy',ylim=(0,1));ax.legend(fontsize=7);fig.tight_layout();fig.savefig(output/(weighting+'-'+risk+'.png'),dpi=160);plt.close(fig)
            fig,axes=plt.subplots(2,4,figsize=(13,7))
            for label,ax in zip(CLASSES,axes.flat):
                for mode in predictions:
                    bins=result['metrics'][mode][weighting]['calibration']['per_class'][label]['bins'];points=[b for b in bins if b['n']]
                    ax.plot([b['mean_probability'] for b in points],[b['positive_fraction'] for b in points],marker='o',label=mode)
                ax.plot([0,1],[0,1],'k--',linewidth=.7);ax.set(title=label,xlim=(0,1),ylim=(0,1));ax.legend(fontsize=6)
            fig.suptitle('Held-out reliability diagrams: '+weighting+'; ten fixed bins, sparse classes limited');fig.tight_layout();fig.savefig(output/(weighting+'-reliability-diagrams.png'),dpi=160);plt.close(fig)
        (output/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        report.update(status='complete',patients=len(raw),images=len(rows),components=len(set(d['component_ids'])),primary_estimate=estimates[primary_key],primary_interval=intervals[primary_key],temperature_reused=temperature,normalizer_reused_sha256=contract['quality_normalizer_sha256'],all_binary_decisions_unchanged=True,bootstrap_replicates=cfg['bootstrap_replicates'],bootstrap_seed=cfg['bootstrap_seed'],automatic_acceptance_policy_fitted=False,packages={'python':platform.python_version(),'numpy':np.__version__,'scipy':__import__('scipy').__version__,'sklearn':__import__('sklearn').__version__})
    except Exception as error:
        import traceback;traceback.print_exc();report.update(error_type=type(error).__name__,error=str(error))
    report['elapsed_seconds']=time.monotonic()-start;finish(output,report);progress('phase7_analysis_finished',status=report['status']);return report
