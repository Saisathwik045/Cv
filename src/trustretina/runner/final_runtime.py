"""One frozen final-only GPU inference; no training, fitting or final-case selection."""
import json,time,platform,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from trustretina.models.backbones import create_backbone
from trustretina.runner.full_runtime import CachedEyes
from trustretina.runner.reliability_runtime import state_hash
from trustretina.reliability.stochastic import stochastic_inventory,head_patient_samples
from trustretina.evaluation.metrics import aggregate_patient_mean
from trustretina.data.validation import file_hash
from .resumed_review import finish,progress


def run(spec,*,input_root,output,nonce,source_commit):
    start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False);cfg=spec['config'];contract=spec['contract']
    report={'stage':'phase7_inference','nonce':nonce,'source_commit':source_commit,'training':False,'gpu_requested':True,'status':'incomplete','protocol_sha256':spec['protocol_sha256'],'candidate_contract_sha256':spec['candidate_contract_sha256'],'parameter_fitting':False,'final_evaluation_access':True,'checkpoint_sha256':spec['checkpoint_sha256']}
    try:
        if not cfg['final_evaluation_authorized'] or cfg['role']!='final_evaluation' or cfg['parameter_fitting'] or cfg['backbone_training'] or cfg['risk_policy_fitting']:raise ValueError('Final inference scope violation')
        candidates=[p for p in Path(input_root).rglob('inputs.json') if file_hash(p)==spec['inputs_sha256']]
        if len(candidates)!=1:raise ValueError('One verified final input index required')
        root=candidates[0].parent;index=json.loads(candidates[0].read_text())
        if index['role']!='final_evaluation' or index['manifest_sha256']!=spec['final_manifest_sha256'] or index['candidate_contract_sha256']!=spec['candidate_contract_sha256'] or index['preprocessing_sha256']!=contract['preprocessing_sha256']:raise ValueError('Final input identity changed')
        for key in ('images','rows','checkpoint'):
            p=(root/index[key+'_file']).resolve()
            if not p.is_relative_to(root.resolve()) or file_hash(p)!=index[key+'_sha256']:raise ValueError('Final input/checkpoint bytes changed')
        if index['checkpoint_sha256']!=spec['checkpoint_sha256']:raise ValueError('Selected checkpoint identity changed')
        if not torch.cuda.is_available():raise RuntimeError('Authorized free GPU unavailable')
        device=torch.device('cuda:0');torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
        deps={'python':platform.python_version(),'torch':str(torch.__version__),'torchvision':__import__('torchvision').__version__,'numpy':np.__version__,'Pillow':__import__('PIL').__version__,'sklearn':__import__('sklearn').__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
        if deps!=spec['expected_dependencies']:raise ValueError('Frozen inference environment changed')
        record=torch.load(root/index['checkpoint_file'],map_location='cpu',weights_only=True)
        if record['provenance']!=spec['selected_provenance'] or record['completed_epoch']!=1:raise ValueError('Wrong selected checkpoint provenance/epoch')
        model=create_backbone('convnext_tiny');model.load_state_dict(record['model'],strict=True);del record;model.to(device);model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        before=state_hash(model);rows=json.loads((root/index['rows_file']).read_text());data=CachedEyes(root/index['images_file'],rows);loader=DataLoader(data,batch_size=16,shuffle=False,num_workers=0)
        features=[];probability=[];direct_error=0.;torch.cuda.synchronize();started=time.monotonic()
        with torch.inference_mode():
            for i,(images,_) in enumerate(loader):
                if time.monotonic()-start>cfg['maximum_gpu_seconds']-120:raise TimeoutError('Final GPU budget')
                images=images.to(device);f=model.backbone(images);p=model.head(f).sigmoid()
                if i==0:direct_error=float((model(images).sigmoid()-p).abs().max().cpu())
                features.append(f);probability.append(p.cpu().numpy())
        torch.cuda.synchronize();det_seconds=time.monotonic()-started
        if direct_error>1e-6:raise ValueError('Direct/feature predictor disagreement')
        features=torch.cat(features);p=np.concatenate(probability);y=np.asarray([r['labels'] for r in rows]);patient=aggregate_patient_mean([r['patient_id'] for r in rows],[r['eye'] for r in rows],y,p,[r['component_id'] for r in rows])
        if len(patient['patient_ids'])!=cfg['expected_patients']:raise ValueError('Incomplete final patient coverage')
        np.save(output/'feature-matrix.npy',features.cpu().numpy());np.savez_compressed(output/'deterministic.npz',labels=patient['labels'],probabilities=patient['probabilities'],patient_ids=patient['patient_ids'],component_ids=patient['component_ids'],image_probabilities=p,image_labels=y,image_patient_ids=[r['patient_id'] for r in rows],image_eyes=[r['eye'] for r in rows],image_ids=[r['path'] for r in rows])
        report.update(completed_stage='deterministic',patients=len(patient['patient_ids']),images=len(rows));finish(output,report)
        mc=head_patient_samples(model,features,rows,passes=contract['mc_passes'],base_seed=cfg['mc_seed'])
        if mc['patient_ids']!=patient['patient_ids'] or mc['seeds']!=list(range(contract['mc_validation_seed'],contract['mc_validation_seed']+20)):raise ValueError('Frozen MC sequence/patient alignment changed')
        if not np.isfinite(mc['probabilities']).all():raise ValueError('Nonfinite stochastic outputs')
        np.savez_compressed(output/'stochastic.npz',patient_ids=patient['patient_ids'],probabilities=mc['probabilities'],seeds=mc['seeds'],mean=mc['probabilities'].mean(0),variance=mc['probabilities'].var(0,ddof=0))
        after=state_hash(model)
        if after!=before or any(m.training for m in model.modules()):raise ValueError('Frozen model state or evaluation modes changed')
        (output/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        report.update(status='complete',completed_stage='deterministic_and_stochastic',dependencies=deps,inventory=stochastic_inventory(model),gpu=torch.cuda.get_device_name(0),state_before_sha256=before,state_after_sha256=after,model_state_unchanged=True,strict_checkpoint_reload=True,mc_passes=20,mc_seeds=mc['seeds'],direct_feature_max_probability_error=direct_error,deterministic_seconds=det_seconds,batch_size=16,case_selection='none; all frozen final patients')
    except Exception as error:
        import traceback;traceback.print_exc();report.update(error_type=type(error).__name__,error=str(error))
    report['elapsed_seconds']=time.monotonic()-start;finish(output,report);progress('phase7_inference_finished',status=report['status']);return report
