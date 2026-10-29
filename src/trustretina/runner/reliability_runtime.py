"""Bounded inference/Grad-CAM only; selected checkpoint immutable and final data absent."""
import hashlib,json,time,subprocess,sys,platform
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from trustretina.models.backbones import create_backbone
from trustretina.runner.full_runtime import CachedEyes
from trustretina.data.validation import file_hash
from trustretina.evaluation.metrics import aggregate_patient_mean
from trustretina.reliability.stochastic import stochastic_inventory,head_patient_samples
from trustretina.gradcam.implementation import patient_gradcam
from .resumed_review import finish,progress


def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        array=value.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(array.dtype).encode());h.update(str(array.shape).encode());h.update(array.tobytes())
    return h.hexdigest()


def run(spec,*,input_root,output,nonce,source_commit):
    start=time.monotonic();output=Path(output);output.mkdir(exist_ok=False,parents=True);cfg=spec['config']
    report={'nonce':nonce,'source_commit':source_commit,'training':False,'gpu_requested':True,'stage':'phase6_inference','status':'incomplete','protocol_sha256':spec['protocol_sha256'],'checkpoint_sha256':spec['checkpoint_sha256'],'release_sha256':spec['release_sha256'],'final_evaluation_access':False,'calibration_fitted':False,'risk_policy_fitted':False}
    try:
        if cfg['final_evaluation_access'] or cfg['backbone_retraining'] or cfg['risk_policy_fitting']:raise ValueError('Scope violation')
        matches=[p for p in Path(input_root).rglob('inputs.json') if file_hash(p)==spec['inputs_sha256']]
        checkpoints=[p for p in Path(input_root).rglob('best.pt') if file_hash(p)==cfg['checkpoint_sha256']]
        if len(matches)!=1 or len(checkpoints)!=1:raise ValueError('Expected one verified input index and checkpoint')
        root=matches[0].parent;index=json.loads(matches[0].read_text())
        if set(index['roles'])!={'validation','calibration'} or index['protocol_sha256']!=spec['protocol_sha256'] or index['release_sha256']!=cfg['release_sha256']:raise ValueError('Permitted inference role/protocol mismatch')
        if not torch.cuda.is_available():raise RuntimeError('Authorized GPU unavailable')
        device=torch.device('cuda:0');torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
        dependencies={'python':platform.python_version(),'torch':str(torch.__version__),'torchvision':__import__('torchvision').__version__,'numpy':np.__version__,'Pillow':__import__('PIL').__version__,'sklearn':__import__('sklearn').__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
        if dependencies!=spec['expected_dependencies']:raise ValueError('Selected checkpoint inference runtime changed')
        (output/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        record=torch.load(checkpoints[0],map_location='cpu',weights_only=True)
        if record['provenance']!=spec['selected_provenance'] or record['completed_epoch']!=1:raise ValueError('Selected epoch/source provenance changed')
        model=create_backbone('convnext_tiny');model.load_state_dict(record['model'],strict=True);del record;model.to(device);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        before=state_hash(model);report.update(dependencies=dependencies,gpu=torch.cuda.get_device_name(0),inventory=stochastic_inventory(model),state_before_sha256=before,strict_checkpoint_reload=True)
        seen={};cam_records=[]
        for role,item in index['roles'].items():
            if time.monotonic()-start>cfg['maximum_gpu_seconds']-120:raise TimeoutError('Bounded inference budget')
            for key in ('images','rows'):
                path=root/item[key+'_file']
                if not path.resolve().is_relative_to(root.resolve()) or file_hash(path)!=item[key+'_sha256']:raise ValueError('Inference input checksum changed')
            rows=json.loads((root/item['rows_file']).read_text());seen[role]=({r['patient_id'] for r in rows},{r['component_id'] for r in rows})
            data=CachedEyes(root/item['images_file'],rows);loader=DataLoader(data,batch_size=16,shuffle=False,num_workers=0)
            features=[];eye=[];direct_error=0.
            with torch.inference_mode():
                for i,(x,_) in enumerate(loader):
                    x=x.to(device);f=model.backbone(x);p=model.head(f).sigmoid()
                    if i==0:direct_error=float((model(x).sigmoid()-p).abs().max().cpu())
                    features.append(f);eye.append(p.cpu().numpy())
            if direct_error>1e-6:raise ValueError('Cached-feature and direct predictors disagree')
            features=torch.cat(features);p=np.concatenate(eye);y=np.asarray([r['labels'] for r in rows]);patient=aggregate_patient_mean([r['patient_id'] for r in rows],[r['eye'] for r in rows],y,p,[r['component_id'] for r in rows])
            mc=head_patient_samples(model,features,rows,passes=cfg['mc_passes'],base_seed=cfg['mc_'+role+'_seed'])
            if mc['patient_ids']!=patient['patient_ids']:raise ValueError('Deterministic/MC patient mismatch')
            if not np.isfinite(mc['probabilities']).all() or mc['probabilities'].var(0).max()<=0:raise ValueError('No finite stochastic variation; do not claim MC uncertainty')
            np.savez_compressed(output/(role+'.npz'),labels=patient['labels'],probabilities=patient['probabilities'],patient_ids=patient['patient_ids'],component_ids=patient['component_ids'],image_probabilities=p,image_patient_ids=[r['patient_id'] for r in rows],image_eyes=[r['eye'] for r in rows],image_ids=[r['path'] for r in rows],mc_patient_probabilities=mc['probabilities'],mc_seeds=mc['seeds'])
            progress('phase6_role_saved',role=role,patients=len(patient['patient_ids']),mc_passes=mc['passes'],direct_feature_max_error=direct_error)
            if role=='validation':
                selected=sorted(patient['patient_ids'],key=lambda pid:hashlib.sha256((str(cfg['gradcam_seed'])+'|'+pid).encode()).hexdigest())[:cfg['gradcam_patients']]
                for case,pid in enumerate(selected):
                    indices=[i for i,r in enumerate(rows) if r['patient_id']==pid];images=torch.stack([data[i][0] for i in indices]).to(device)
                    with torch.enable_grad():cam=patient_gradcam(model,images)
                    if not np.isfinite(cam['heatmaps']).all():raise ValueError('Nonfinite explanation map')
                    np.savez_compressed(output/f'gradcam-{case}.npz',heatmaps=cam['heatmaps'],images=np.stack([data.images[i] for i in indices]),patient_probabilities=cam['patient_probabilities'])
                    for j,i in enumerate(indices):
                        rgb=data.images[i].transpose(1,2,0).astype(float);heat=cam['heatmaps'][j];color=np.stack([255*heat,np.zeros_like(heat),255*(1-heat)],axis=2);overlay=np.clip(.65*rgb+.35*color,0,255).astype(np.uint8)
                        Image.fromarray(overlay).save(output/f'gradcam-{case}-{rows[i]["eye"]}.png')
                    cam_records.append({'case':case,'patient_id':pid,'eyes':[rows[i]['eye'] for i in indices],'image_ids':[rows[i]['path'] for i in indices],'raw_image_sha256':[rows[i]['raw_sha256'] for i in indices],'preprocessing_sha256':index['preprocessing_sha256'],'geometry':'canonical EXIF/RGB 224 letterbox; overlays on model input, not inverse-mapped raw pixels','layer':'backbone.features.7','target_index':cam['target_index'],'positive_indices':cam['positive_indices'],'target_below_threshold':cam['target_below_threshold'],'zero_maps':cam['zero_maps'],'score':cam['score'],'interpretation':cam['interpretation']})
            del features
        if seen['validation'][0]&seen['calibration'][0] or seen['validation'][1]&seen['calibration'][1]:raise ValueError('Validation/calibration leakage')
        (output/'gradcam-records.json').write_text(json.dumps(cam_records,indent=2));after=state_hash(model)
        if before!=after:raise ValueError('Model parameters/buffers changed during inference')
        report.update(status='complete',state_after_sha256=after,model_state_unchanged=True,roles={role:{'patients':len(v[0]),'groups':len(v[1])} for role,v in seen.items()},mc_passes=cfg['mc_passes'],gradcam_cases=len(cam_records),gradcam_maps=2*len(cam_records),gradcam_zero_maps=sum(sum(x['zero_maps']) for x in cam_records),normalization_and_stochastic_depth_evaluation_mode=True)
    except Exception as e:
        import traceback;traceback.print_exc();report.update(error_type=type(e).__name__,error=str(e))
    report['elapsed_seconds']=time.monotonic()-start;finish(output,report);progress('phase6_inference_finished',status=report['status']);return report
