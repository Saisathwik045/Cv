"""One free-Kaggle full run. Only the verified train/validation cache is mounted."""
import json, os, random, shutil, subprocess, sys, time, platform
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from trustretina.data.validation import file_hash
from trustretina.preprocessing.shared import PreprocessConfig
from trustretina.models.backbones import create_backbone
from trustretina.evaluation.metrics import aggregate_patient_mean
from trustretina.training.checkpoint import save_checkpoint, load_checkpoint
from trustretina.training.comparison import training_weights, patient_loss, advance_selection, comparison_metrics
from .resumed_review import finish, progress


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


class CachedEyes(Dataset):
    def __init__(self, images, rows):
        self.images = np.load(images, mmap_mode='r', allow_pickle=False)
        self.rows = rows
        if self.images.dtype != np.uint8 or self.images.shape != (len(rows),3,224,224):
            raise ValueError('Unexpected cached image tensor')
        self.mean = np.asarray(PreprocessConfig().mean,np.float32)[:,None,None]
        self.std = np.asarray(PreprocessConfig().std,np.float32)[:,None,None]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        x = (self.images[i].astype(np.float32)/255-self.mean)/self.std
        return torch.from_numpy(x), torch.tensor(self.rows[i]['labels'],dtype=torch.float32)


def evaluate(model, loader, rows, device, weights):
    model.eval(); values=[]
    torch.cuda.synchronize(); start=time.monotonic()
    with torch.inference_mode():
        for x, _ in loader:
            values.append(model(x.to(device)).sigmoid().cpu().numpy())
    torch.cuda.synchronize(); seconds=time.monotonic()-start
    p=np.concatenate(values); y=np.asarray([r['labels'] for r in rows])
    if not np.isfinite(p).all(): raise ValueError('Non-finite validation probabilities')
    patient=aggregate_patient_mean([r['patient_id'] for r in rows],[r['eye'] for r in rows],y,p,[r['component_id'] for r in rows])
    return y,p,patient,patient_loss(patient['labels'],patient['probabilities'],weights),seconds


def run(spec, *, input_root, output, nonce, source_commit):
    started=time.monotonic(); output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg=spec['config'];name=spec['backbone'];history=[];selection={};global_step=0;epoch_start=0
    report={'nonce':nonce,'source_commit':source_commit,'stage':'full_v5_training','training':True,'gpu_requested':True,'status':'incomplete','backbone':name,'config_sha256':spec['config_sha256'],'release_sha256':cfg['release_manifest_sha256'],'final_evaluation_access':False,'calibration_fitted':False,'risk_policy_fitted':False,'initialization':'IMAGENET1K_V1','history':history}
    try:
        atomic_json(output/'experiment-spec.json',spec)
        if name not in cfg['backbones'] or cfg['max_epochs']>10 or cfg['final_evaluation_access'] or cfg['calibration_fitting'] or cfg['risk_policy_fitting']:
            raise ValueError('Experiment authorization mismatch')
        # No dataset source may be attached to this job. Only private kernel outputs.
        if spec.get('dataset_sources') != []: raise ValueError('Raw/final data must not be mounted')
        candidates=[p for p in Path(input_root).rglob('inputs.json') if file_hash(p)==spec['inputs_sha256']]
        if len(candidates)!=1:raise ValueError('Verified shared input cache missing/ambiguous')
        root=candidates[0].parent;inputs=json.loads(candidates[0].read_text())
        if inputs['release_sha256']!=cfg['release_manifest_sha256'] or inputs['config_sha256']!=spec['config_sha256'] or inputs['preprocessing_sha256']!=PreprocessConfig().fingerprint() or set(inputs['roles'])!={'train','validation'}:
            raise ValueError('Input configuration mismatch')
        rows={};datasets={}
        for role,item in inputs['roles'].items():
            for kind in ('images','rows'):
                p=root/item[kind+'_file']
                if not p.resolve().is_relative_to(root.resolve()) or file_hash(p)!=item[kind+'_sha256']:raise ValueError('Input checksum mismatch')
            rows[role]=json.loads((root/item['rows_file']).read_text())
            datasets[role]=CachedEyes(root/item['images_file'],rows[role])
        for key in ('patient_id','component_id'):
            if {r[key] for r in rows['train']} & {r[key] for r in rows['validation']}:raise ValueError('Train/validation leakage')
        if not torch.cuda.is_available():raise RuntimeError('Authorized GPU unavailable; no CPU substitute')
        device=torch.device('cuda:0');torch.set_num_threads(2)
        seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        generator=torch.Generator().manual_seed(seed)
        train=DataLoader(datasets['train'],batch_size=cfg['batch_size'],shuffle=True,generator=generator,num_workers=0)
        val=DataLoader(datasets['validation'],batch_size=cfg['batch_size'],shuffle=False,num_workers=0)
        weights=training_weights(rows['train']);w=inputs['weights'][name]
        model=create_backbone(name,weights_path=root/w['file'],expected_weights_sha256=w['sha256']).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights,dtype=torch.float32,device=device))
        provenance={'backbone':name,'split_sha256':cfg['release_manifest_sha256'],'config_sha256':spec['config_sha256'],'preprocessing_sha256':inputs['preprocessing_sha256'],'seed':seed,'data_kind':'real_odir_full_v5','inputs_sha256':spec['inputs_sha256'],'initialization_sha256':w['sha256'],'experiment_source_sha256':spec['experiment_source_sha256']}
        dependencies={'python':platform.python_version(),'torch':str(torch.__version__),'torchvision':__import__('torchvision').__version__,'numpy':np.__version__,'Pillow':__import__('PIL').__version__,'sklearn':__import__('sklearn').__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
        (output/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        report.update(provenance=provenance,dependencies=dependencies,gpu=torch.cuda.get_device_name(0),positive_weights=weights.tolist(),parameter_count=sum(p.numel() for p in model.parameters()),initialization_record=w)
        if spec.get('resume'):
            prior=spec['resume'];matches=[p for p in Path(input_root).rglob('progress.json') if file_hash(p)==prior['progress_sha256']]
            if len(matches)!=1:raise ValueError('Resume progress missing/ambiguous')
            old=matches[0].parent;state=json.loads(matches[0].read_text())
            if state['provenance']!=provenance:raise ValueError('Incompatible resume provenance')
            for filename in ('best.pt','last.pt','history.json'):
                if file_hash(old/filename)!=state['hashes'][filename]:raise ValueError('Resume artifact mismatch')
                shutil.copyfile(old/filename,output/filename)
            loaded=load_checkpoint(output/'last.pt',model,optimizer,expected_sha256=state['hashes']['last.pt'],expected_provenance=provenance,generator=generator)
            epoch_start=loaded['next_epoch'];global_step=loaded['global_step'];selection=state['selection'];history=json.loads((output/'history.json').read_text())
            report.update(resumed_from=prior,history=history)
        complete=False
        for epoch in range(epoch_start,cfg['max_epochs']):
            if selection.get('early_stopped'):complete=True;break
            # Stop at epoch boundaries with room for durable checkpoint/report writes.
            if time.monotonic()-started>cfg['maximum_gpu_session_seconds']-600:break
            epoch_time=time.monotonic();model.train();total=0.;seen=0
            for x,y in train:
                x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True)
                loss=criterion(model(x),y)
                if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
                loss.backward()
                norms=torch._foreach_norm([p.grad for p in model.parameters() if p.grad is not None])
                if not torch.isfinite(torch.stack(norms)).all():raise ValueError('Non-finite gradient')
                optimizer.step();global_step+=1;total+=loss.item()*len(y);seen+=len(y)
            _,_,patient,vl,vs=evaluate(model,val,rows['validation'],device,weights)
            selection,improved=advance_selection(selection,vl,epoch,min_delta=cfg['early_stopping_min_delta'],patience=cfg['early_stopping_patience'])
            history.append({'epoch':epoch+1,'training_image_weighted_bce':total/seen,'validation_patient_weighted_bce':vl,'validation_patient_macro_auroc':comparison_metrics(patient['labels'],patient['probabilities'])['macro_auroc'],'epoch_seconds':time.monotonic()-epoch_time,'validation_seconds':vs,'global_step':global_step})
            args=dict(completed_epoch=epoch,global_step=global_step,provenance=provenance,generator=generator,best_score=selection['best_loss'])
            if improved:save_checkpoint(output/'best.pt',model,optimizer,**args)
            save_checkpoint(output/'last.pt',model,optimizer,**args)
            atomic_json(output/'history.json',history)
            atomic_json(output/'progress.json',{'provenance':provenance,'selection':selection,'completed_epoch':epoch,'global_step':global_step,'hashes':{f:file_hash(output/f) for f in ('best.pt','last.pt','history.json')}})
            complete=selection['early_stopped'] or epoch+1==cfg['max_epochs']
            report.update(status='complete' if complete else 'paused',history=history,selection=selection,completed_epochs=epoch+1,elapsed_seconds=time.monotonic()-started)
            finish(output,report);progress('full_epoch_saved',backbone=name,epoch=epoch+1,validation_patient_loss=vl,early_stopped=selection['early_stopped'])
            if complete:break
        if not (output/'best.pt').exists():raise RuntimeError('Session ended before first epoch checkpoint')
        state=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
        if state['provenance']!=provenance:raise ValueError('Best checkpoint provenance mismatch')
        model.load_state_dict(state['model'],strict=True)
        # Two warm-up batches excluded; measured pass includes data transfer and forward only.
        model.eval()
        with torch.inference_mode():
            for i,(x,_) in enumerate(val):
                model(x.to(device))
                if i==1:break
        y,p,patient,_,seconds=evaluate(model,val,rows['validation'],device,weights)
        np.savez_compressed(output/'validation-images.npz',labels=y,probabilities=p,patient_ids=np.asarray([r['patient_id'] for r in rows['validation']]),eyes=np.asarray([r['eye'] for r in rows['validation']]),image_ids=np.asarray([r['path'] for r in rows['validation']]),component_ids=np.asarray([r['component_id'] for r in rows['validation']]))
        np.savez_compressed(output/'validation-patients.npz',labels=patient['labels'],probabilities=patient['probabilities'],patient_ids=np.asarray(patient['patient_ids']),component_ids=np.asarray(patient['component_ids']))
        metrics={'image':comparison_metrics(y,p),'patient':comparison_metrics(patient['labels'],patient['probabilities']),'inference':{'seconds':seconds,'images':len(y),'milliseconds_per_image':1000*seconds/len(y),'batch_size':cfg['batch_size'],'definition':'warm cached end-to-end validation transfer/forward/sigmoid/CPU collection; excludes original decode/resize and metric computation; two warm-up batches excluded'},'parameter_count':report['parameter_count'],'patient_aggregation':patient['aggregation']}
        atomic_json(output/'metrics.json',metrics)
        report.update(status='complete' if complete else 'paused',strict_best_reload=True,metrics=metrics,history=history,selection=selection)
    except Exception as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e))
        import traceback;traceback.print_exc()
    report.update(elapsed_seconds=time.monotonic()-started)
    finish(output,report);progress('full_run_finished',backbone=name,status=report['status'])
    return report
