"""V11 training-only worker: inner selection, deferred outer assessment, resumable."""
import json, os, random, shutil, subprocess, sys, time, platform
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from trustretina.data.validation import file_hash
from trustretina.preprocessing.shared import PreprocessConfig
from trustretina.models.backbones import create_backbone
from trustretina.evaluation.metrics import aggregate_patient_mean
from trustretina.training.checkpoint import save_checkpoint, load_checkpoint
from trustretina.training.comparison import patient_loss, comparison_metrics
from .resumed_review import finish, progress
from trustretina.training.controlled import advance_auc
from trustretina.training.nested import validate_config,audit_manifest,loss_weights,MATRIX,outer_after_freeze


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
    cfg=spec['config'];name=cfg['backbone'];history=[];selection={};global_step=0;epoch_start=0
    report={'nonce':nonce,'source_commit':source_commit,'stage':'nested_v11_training','training':True,'gpu_requested':True,'status':'incomplete','backbone':name,'config_sha256':spec['config_sha256'],'release_sha256':spec['release_sha256'],'final_evaluation_access':False,'calibration_fitted':False,'risk_policy_fitted':False,'initialization':'IMAGENET1K_V1','history':history}
    try:
        atomic_json(output/'experiment-spec.json',spec)
        validate_config(cfg)
        if (spec['run_id'],spec['outer_fold'],spec['arm']) not in MATRIX:raise ValueError('Run not in fixed matrix')
        if spec.get('execution_approved') is not True:raise PermissionError('Worker execution not approved')
        report.update(source_role='train',arm=spec['arm'],outer_fold=spec['outer_fold'],outer_assessed=False,plan_sha256=spec['plan_sha256'],protocol_sha256=spec['protocol_sha256'],run_id=spec['run_id'])
        if name!='convnext_tiny' or cfg['max_epochs']>25 or cfg['historical_final_access'] or cfg['calibration_fitting'] or cfg['risk_policy_fitting']:
            raise ValueError('Experiment authorization mismatch')
        # No dataset source may be attached to this job. Only private kernel outputs.
        if spec.get('dataset_sources') != []: raise ValueError('Raw/final data must not be mounted')
        candidates=[p for p in Path(input_root).rglob('inputs.json') if file_hash(p)==spec['inputs_sha256']]
        if len(candidates)!=1:raise ValueError('Verified shared input cache missing/ambiguous')
        root=candidates[0].parent;inputs=json.loads(candidates[0].read_text())
        if inputs['release_sha256']!=spec['release_sha256'] or inputs['config_sha256']!=spec['cache_parent_config_sha256'] or inputs['preprocessing_sha256']!=PreprocessConfig().fingerprint() or set(inputs['roles'])!={'train','validation'}:
            raise ValueError('Input configuration mismatch')
        # Open only the TRAIN cache; old validation payloads are never loaded.
        item=inputs['roles']['train']
        for kind in ('images','rows'):
            p=root/item[kind+'_file']
            if not p.resolve().is_relative_to(root.resolve()) or file_hash(p)!=item[kind+'_sha256']:raise ValueError('Train input checksum mismatch')
        all_rows=json.loads((root/item['rows_file']).read_text())
        manifest=spec['fold_manifest']
        if manifest['sha256']!=spec['fold_manifest_sha256'] or manifest['train_rows_sha256']!=item['rows_sha256']:
            raise ValueError('Private fold identity mismatch')
        audit_manifest(all_rows,manifest,cfg)
        fold=manifest['folds'][spec['outer_fold']]
        base=CachedEyes(root/item['images_file'],all_rows)
        rows={r:[all_rows[i] for i in fold[r+'_indices']] for r in ('fit','inner')}
        datasets={r:Subset(base,fold[r+'_indices']) for r in ('fit','inner')}
        if not torch.cuda.is_available():raise RuntimeError('Authorized GPU unavailable; no CPU substitute')
        device=torch.device('cuda:0');torch.set_num_threads(2)
        seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        generator=torch.Generator().manual_seed(seed)
        train=DataLoader(datasets['fit'],batch_size=cfg['batch_size'],shuffle=True,generator=generator,num_workers=0)
        val=DataLoader(datasets['inner'],batch_size=cfg['batch_size'],shuffle=False,num_workers=0)
        weights=loss_weights(rows['fit'],spec['arm']);w=inputs['weights'][name]
        model=create_backbone(name,weights_path=root/w['file'],expected_weights_sha256=w['sha256']).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['max_epochs'],eta_min=cfg['minimum_learning_rate'])
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights,dtype=torch.float32,device=device))
        provenance={'backbone':name,'split_sha256':spec['release_sha256'],'config_sha256':spec['config_sha256'],'preprocessing_sha256':inputs['preprocessing_sha256'],'seed':seed,'data_kind':'real_odir_training_only_nested_v11','plan_sha256':spec['plan_sha256'],'protocol_sha256':spec['protocol_sha256'],'run_id':spec['run_id'],'schedule':'cosine','arm':spec['arm'],'outer_fold':spec['outer_fold'],'fold_manifest_sha256':spec['fold_manifest_sha256'],'inputs_sha256':spec['inputs_sha256'],'initialization_sha256':w['sha256'],'experiment_source_sha256':spec['experiment_source_sha256']}
        dependencies={'python':platform.python_version(),'torch':str(torch.__version__),'torchvision':__import__('torchvision').__version__,'numpy':np.__version__,'Pillow':__import__('PIL').__version__,'sklearn':__import__('sklearn').__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
        if dependencies!=spec['expected_dependencies']:raise ValueError('Pinned training environment changed')
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
            loaded=load_checkpoint(output/'last.pt',model,optimizer,expected_sha256=state['hashes']['last.pt'],expected_provenance=provenance,generator=generator,scheduler=scheduler)
            epoch_start=loaded['next_epoch'];global_step=loaded['global_step'];selection=state['selection'];history=json.loads((output/'history.json').read_text())
            report.update(resumed_from=prior,history=history)
        complete=False
        for epoch in range(epoch_start,cfg['max_epochs']):
            if selection.get('early_stopped'):complete=True;break
            # Stop at epoch boundaries with room for durable checkpoint/report writes.
            if time.monotonic()-started>cfg['maximum_gpu_session_seconds']-180:break
            epoch_time=time.monotonic();epoch_lr=optimizer.param_groups[0]['lr'];model.train();total=0.;seen=0
            for x,y in train:
                x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True)
                loss=criterion(model(x),y)
                if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
                loss.backward()
                norms=torch._foreach_norm([p.grad for p in model.parameters() if p.grad is not None])
                if not torch.isfinite(torch.stack(norms)).all():raise ValueError('Non-finite gradient')
                optimizer.step();global_step+=1;total+=loss.item()*len(y);seen+=len(y)
            _,_,patient,vl,vs=evaluate(model,val,rows['inner'],device,weights)
            validation_metrics=comparison_metrics(patient['labels'],patient['probabilities'])
            selection,improved=advance_auc(selection,validation_metrics['macro_auroc'],epoch,min_delta=cfg['early_stopping_min_delta'],patience=cfg['early_stopping_patience'])
            if scheduler is not None:scheduler.step()
            history.append({'epoch':epoch+1,'learning_rate':epoch_lr,'next_learning_rate':optimizer.param_groups[0]['lr'],'inner_patient_metrics':validation_metrics,'training_image_weighted_bce':total/seen,'inner_patient_weighted_bce':vl,'inner_patient_macro_auroc':validation_metrics['macro_auroc'],'epoch_seconds':time.monotonic()-epoch_time,'inner_selection_seconds':vs,'global_step':global_step})
            args=dict(completed_epoch=epoch,global_step=global_step,provenance=provenance,generator=generator,best_score=selection['best_score'],scheduler=scheduler)
            if improved:save_checkpoint(output/'best.pt',model,optimizer,**args)
            save_checkpoint(output/'last.pt',model,optimizer,**args)
            atomic_json(output/'history.json',history)
            atomic_json(output/'progress.json',{'provenance':provenance,'selection':selection,'completed_epoch':epoch,'global_step':global_step,'hashes':{f:file_hash(output/f) for f in ('best.pt','last.pt','history.json')}})
            complete=selection['early_stopped'] or epoch+1==cfg['max_epochs']
            report.update(status='complete' if complete else 'paused',history=history,selection=selection,completed_epochs=epoch+1,elapsed_seconds=time.monotonic()-started)
            finish(output,report);progress('full_epoch_saved',backbone=name,epoch=epoch+1,inner_patient_loss=vl,early_stopped=selection['early_stopped'])
            if complete:break
        if not (output/'best.pt').exists():raise RuntimeError('Session ended before first epoch checkpoint')
        if not complete:
            report.update(status='paused',outer_assessed=False,elapsed_seconds=time.monotonic()-started)
            finish(output,report);return report
        state=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
        if state['provenance']!=provenance:raise ValueError('Best checkpoint provenance mismatch')
        model.load_state_dict(state['model'],strict=True)
        report['strict_best_reload']=True
        def frozen_outer_pass():
            rows['outer']=[all_rows[i] for i in fold['outer_indices']]
            outer_loader=DataLoader(Subset(base,fold['outer_indices']),batch_size=cfg['batch_size'],shuffle=False,num_workers=0)
            return evaluate(model,outer_loader,rows['outer'],device,weights)
        y,p,patient,_,seconds=outer_after_freeze(report,complete,frozen_outer_pass)
        np.savez_compressed(output/'outer-images.npz',labels=y,probabilities=p,patient_ids=np.asarray([r['patient_id'] for r in rows['outer']]),eyes=np.asarray([r['eye'] for r in rows['outer']]),image_ids=np.asarray([r['path'] for r in rows['outer']]),component_ids=np.asarray([r['component_id'] for r in rows['outer']]))
        np.savez_compressed(output/'outer-patients.npz',labels=patient['labels'],probabilities=patient['probabilities'],patient_ids=np.asarray(patient['patient_ids']),component_ids=np.asarray(patient['component_ids']))
        metrics={'image':comparison_metrics(y,p),'patient':comparison_metrics(patient['labels'],patient['probabilities']),'inference':{'seconds':seconds,'images':len(y),'milliseconds_per_image':1000*seconds/len(y),'batch_size':cfg['batch_size'],'definition':'single post-freeze outer transfer/forward/sigmoid/CPU collection; excludes original decode/resize and metric computation'},'parameter_count':report['parameter_count'],'patient_aggregation':patient['aggregation']}
        atomic_json(output/'metrics.json',metrics)
        report.update(status='complete' if complete else 'paused',strict_best_reload=True,outer_assessed=True,metrics=metrics,history=history,selection=selection)
    except Exception as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e))
        import traceback;traceback.print_exc()
    report.update(elapsed_seconds=time.monotonic()-started)
    finish(output,report);progress('full_run_finished',backbone=name,status=report['status'])
    return report
