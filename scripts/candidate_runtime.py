"""V14 two-role training worker; actual GPU execution requires a separate exact approval."""
import json,os,random,shutil,sys,time,platform,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset,Dataset
from trustretina.data.validation import file_hash
from trustretina.preprocessing.shared import PreprocessConfig
from trustretina.models.backbones import create_backbone
from trustretina.training.checkpoint import save_checkpoint,load_checkpoint
from trustretina.training.controlled import advance_auc
from trustretina.training.comparison import comparison_metrics
from trustretina.evaluation.threshold_cv import summarize
from trustretina.reliability.stochastic import trained_head_dropout
from trustretina.reliability.development import canonical_quality
from trustretina.runner.resumed_review import finish
from resolution_runtime import CachedEyes,evaluate,atomic_json
from resolution_study import train_accumulated_epoch
from candidate_study import MATRIX,validate_config,audit_roles,train_weights,mean_eyes,calibration_after_freeze


def index_at(root,digest):
    found=[p for p in Path(root).rglob('inputs.json') if file_hash(p)==digest]
    if len(found)!=1:raise ValueError('Exact input index missing/ambiguous')
    return found[0].parent,json.loads(found[0].read_text())


class UnlabeledEyes(Dataset):
    def __init__(self,path,rows,size):
        if any('labels' in r for r in rows):raise ValueError('Calibration labels forbidden')
        self.a=np.load(path,mmap_mode='r',allow_pickle=False);self.rows=rows
        if self.a.dtype!=np.uint8 or self.a.shape!=(len(rows),3,size,size):raise ValueError('Calibration tensor mismatch')
        self.mean=np.asarray(PreprocessConfig().mean,np.float32)[:,None,None];self.std=np.asarray(PreprocessConfig().std,np.float32)[:,None,None]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return torch.from_numpy((self.a[i].astype(np.float32)/255-self.mean)/self.std)


def features_and_predictions(model,path,rows,size,device,batch):
    data=UnlabeledEyes(path,[{k:v for k,v in r.items() if k!='labels'} for r in rows],size)
    features=[];raw_logits=[];model.eval();order=sorted(range(len(rows)),key=lambda i:(rows[i]['patient_id'],rows[i]['eye']));ordered=[rows[i] for i in order]
    with torch.inference_mode():
        for x in DataLoader(Subset(data,order),batch_size=batch,shuffle=False,num_workers=0):
            feature=model.backbone(x.to(device)).detach();features.append(feature);raw_logits.append(model.head(feature))
        f=torch.cat(features);logits=torch.cat(raw_logits);prob=logits.sigmoid().cpu().numpy()
        patient=mean_eyes(ordered,prob)
        samples=[]
        with torch.random.fork_rng(devices=[0] if f.is_cuda else []),trained_head_dropout(model):
            for seed in range(72026 if 'labels' not in rows[0] else 62026,(72026 if 'labels' not in rows[0] else 62026)+20):
                torch.manual_seed(seed)
                if f.is_cuda:torch.cuda.manual_seed_all(seed)
                samples.append(mean_eyes(ordered,model.head(f).sigmoid().cpu().numpy())['probabilities'])
    return logits.cpu().numpy(),prob,patient,np.stack(samples),ordered


def run(spec,*,input_root,output,nonce,source_commit):
    started=time.monotonic();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    report={'nonce':nonce,'source_commit':source_commit,'training':True,'gpu_requested':True,'stage':'candidate_v14_training','status':'incomplete','final_evaluation_access':False,'calibration_fitted':False,'threshold_fitted':False,'risk_policy_fitted':False,'calibration_labels_accessed':False,'history':[]}
    try:
        cfg=validate_config(spec['config']);size=spec['image_size'];history=[];selection={};step=0;first_epoch=0
        if not spec.get('execution_approved') or (spec['run_id'],size) not in MATRIX or spec.get('dataset_sources')!=[]:raise PermissionError('Frozen authorized v14 request required; raw datasets forbidden')
        if spec['session_timeout_seconds'] not in (7500,3900,2100):raise ValueError('Session time cap changed')
        atomic_json(out/'experiment-spec.json',spec)
        trainroot,trainindex=index_at(input_root,spec['train_inputs_sha256']);roleroot,index=index_at(input_root,spec['inputs_sha256'])
        if index['historical_final_included'] or index['calibration_labels_included'] or index['release_sha256']!=spec['release_sha256']:raise ValueError('Input scope mismatch')
        roles={};images={};meta={}
        item=trainindex['resolutions'][str(size)];meta['train']={'rows_file':trainindex['rows_file'],'rows_sha256':trainindex['rows_sha256'],'resolutions':{str(size):item}}
        for role in ('train','validation','calibration'):
            root=trainroot if role=='train' else roleroot;m=meta['train'] if role=='train' else index['roles'][role]
            rowpath=root/m['rows_file'];image=root/m['resolutions'][str(size)]['file']
            if file_hash(rowpath)!=m['rows_sha256'] or file_hash(image)!=m['resolutions'][str(size)]['sha256'] or m['resolutions'][str(size)]['preprocessing_sha256']!=PreprocessConfig(size=size).fingerprint():raise ValueError('Actual role bytes/preprocessing changed')
            roles[role]=json.loads(rowpath.read_text());images[role]=image
            if role!='train' and m['manifest_sha256']!=spec['role_manifest_sha256'][role]:raise ValueError('Role manifest changed')
        report['role_support']=audit_roles(roles,{'train':2275,'validation':525,'calibration':350})
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='Tesla T4':raise RuntimeError('Exactly one free-requested Tesla T4 required; no CPU/multi-GPU substitute')
        deps={'python':platform.python_version(),'torch':str(torch.__version__),'torchvision':__import__('torchvision').__version__,'numpy':np.__version__,'Pillow':__import__('PIL').__version__,'sklearn':__import__('sklearn').__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
        if deps!=spec['expected_dependencies']:raise ValueError('Training environment changed')
        torch.set_num_threads(2);seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
        generator=torch.Generator().manual_seed(seed);device=torch.device('cuda:0')
        base={r:CachedEyes(images[r],roles[r],size) for r in ('train','validation')}
        order=sorted(range(len(roles['validation'])),key=lambda i:(roles['validation'][i]['patient_id'],roles['validation'][i]['eye']))
        valrows=[roles['validation'][i] for i in order]
        train=DataLoader(base['train'],batch_size=8,shuffle=True,generator=generator,num_workers=0)
        val=DataLoader(Subset(base['validation'],order),batch_size=8,shuffle=False,num_workers=0)
        weights=train_weights(roles['train']);w=trainindex['weights']['convnext_tiny']
        model=create_backbone('convnext_tiny',weights_path=trainroot/w['file'],expected_weights_sha256=w['sha256']).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=25,eta_min=cfg['minimum_learning_rate'])
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights,dtype=torch.float32,device=device))
        provenance={'backbone':'convnext_tiny','split_sha256':spec['release_sha256'],'config_sha256':spec['config_sha256'],'preprocessing_sha256':PreprocessConfig(size=size).fingerprint(),'seed':seed,'data_kind':'real_odir_frozen_v4_roles_v14','plan_sha256':spec['plan_sha256'],'run_id':spec['run_id'],'image_size':size,'worker_source_sha256':spec['worker_source_sha256'],'inputs_sha256':spec['inputs_sha256'],'train_inputs_sha256':spec['train_inputs_sha256'],'initialization_sha256':w['sha256']}
        report.update(plan_sha256=spec['plan_sha256'],run_id=spec['run_id'],image_size=size,provenance=provenance,dependencies=deps,gpu='Tesla T4',positive_weights=weights.tolist(),initialization_record=w,history=history)
        (out/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        if spec.get('resume'):
            prior=spec['resume'];matches=[p for p in Path(input_root).rglob('progress.json') if file_hash(p)==prior['progress_sha256']]
            if len(matches)!=1:raise ValueError('Resume progress missing/ambiguous')
            old=matches[0].parent;state=json.loads(matches[0].read_text())
            if state['provenance']!=provenance:raise ValueError('Incompatible resumed fit/input/source')
            for name in ('best.pt','last.pt','history.json'):
                if file_hash(old/name)!=state['hashes'][name]:raise ValueError('Resume output mismatch')
                shutil.copyfile(old/name,out/name)
            restored=load_checkpoint(out/'last.pt',model,optimizer,expected_sha256=state['hashes']['last.pt'],expected_provenance=provenance,generator=generator,scheduler=scheduler)
            first_epoch=restored['next_epoch'];step=restored['global_step'];selection=state['selection'];history=json.loads((out/'history.json').read_text());report.update(history=history,resumed_from=prior)
            if first_epoch!=len(history):raise ValueError('Resume epoch/history boundary differs')
            for filename in ('best.pt','last.pt'):(out/(filename+'.sha256')).write_text(file_hash(out/filename))
        complete=bool(selection.get('early_stopped')) or first_epoch==25
        for epoch in range(first_epoch,25):
            if complete:break
            epoch_budget=max([x['epoch_seconds'] for x in history]+[300 if size==320 else 180])+180
            if time.monotonic()-started+epoch_budget>spec['session_timeout_seconds']:break
            now=time.monotonic();lr=optimizer.param_groups[0]['lr'];model.train()
            loss,steps=train_accumulated_epoch(model,train,optimizer,criterion,device,accumulation_steps=2);step+=steps
            _,_,patient,vl,vs=evaluate(model,val,valrows,device,weights);metrics=comparison_metrics(patient['labels'],patient['probabilities'])
            selection,improved=advance_auc(selection,metrics['macro_auroc'],epoch,min_delta=1e-4,patience=6);scheduler.step()
            history.append({'epoch':epoch+1,'learning_rate':lr,'next_learning_rate':optimizer.param_groups[0]['lr'],'training_image_weighted_bce':loss,'validation_patient_weighted_bce':vl,'validation_patient_macro_auroc':metrics['macro_auroc'],'validation_patient_metrics':metrics,'global_step':step,'epoch_seconds':time.monotonic()-now})
            args=dict(completed_epoch=epoch,global_step=step,provenance=provenance,generator=generator,best_score=selection['best_score'],scheduler=scheduler)
            if improved:save_checkpoint(out/'best.pt',model,optimizer,**args)
            save_checkpoint(out/'last.pt',model,optimizer,**args);atomic_json(out/'history.json',history)
            atomic_json(out/'progress.json',{'provenance':provenance,'selection':selection,'completed_epoch':epoch,'hashes':{f:file_hash(out/f) for f in ('best.pt','last.pt','history.json')}})
            complete=selection['early_stopped'] or epoch+1==25
            report.update(status='complete' if complete else 'paused',history=history,selection=selection,completed_epochs=epoch+1,elapsed_seconds=time.monotonic()-started);finish(out,report)
            print(json.dumps({'epoch':epoch+1,'checkpoint_saved':True,'run_id':spec['run_id']}),flush=True)
        if not (out/'best.pt').exists():raise RuntimeError('No complete epoch checkpoint before session cap')
        if not complete:
            report.update(status='paused',elapsed_seconds=time.monotonic()-started);return finish(out,report)
        state=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        if state['provenance']!=provenance:raise ValueError('Selected checkpoint provenance changed')
        model.load_state_dict(state['model'],strict=True);best_hash=file_hash(out/'best.pt')
        atomic_json(out/'selected-checkpoint.json',{'checkpoint_sha256':best_hash,'best_epoch':selection['best_epoch']+1,'provenance':provenance,'patient_aggregation':cfg['patient_aggregation'],'decision_threshold':.5,'selection_role':'validation'})
        report.update(status='complete',strict_best_reload=True,selected_checkpoint_sha256=best_hash)
        before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        for role in ('validation','calibration'):
            def prediction_pass():return features_and_predictions(model,images[role],roles[role],size,device,8)
            result=calibration_after_freeze(report,prediction_pass) if role=='calibration' else prediction_pass()
            logits,p,patient,passes,ordered=result
            payload={'logits':logits,'probabilities':p,'image_ids':np.asarray([r['path'] for r in ordered]),'patient_ids':np.asarray([r['patient_id'] for r in ordered]),'eyes':np.asarray([r['eye'] for r in ordered])}
            if role=='validation':payload['labels']=np.asarray([r['labels'] for r in ordered])
            if role=='validation':
                targets={r['patient_id']:r['labels'] for r in ordered};patient['labels']=np.asarray([targets[i] for i in patient['patient_ids']])
            np.savez_compressed(out/f'{role}-images.npz',**payload);np.savez_compressed(out/f'{role}-patients.npz',**patient)
            np.savez_compressed(out/f'{role}-passes.npz',probabilities=passes,patient_ids=patient['patient_ids'],seeds=np.arange(62026 if role=='validation' else 72026,(62026 if role=='validation' else 72026)+20))
            if role=='validation':
                targets={r['patient_id']:r['labels'] for r in roles[role]};y=np.asarray([targets[i] for i in patient['patient_ids']]);m=comparison_metrics(y,patient['probabilities']);m['confusion']=summarize(y,patient['probabilities']>=.5)['per_class'];atomic_json(out/'metrics.json',m);report['metrics']=m
        quality={r:np.stack([canonical_quality(x) for x in np.load(images[r],mmap_mode='r',allow_pickle=False)]) for r in ('train','validation','calibration')}
        for role in ('train','validation','calibration'):
            for field in ('path','patient_id','eye','component_id'):quality[role+'_'+field]=np.asarray([r[field] for r in roles[role]])
        np.savez_compressed(out/'quality-features.npz',**quality)
        if any(not torch.equal(before[k],v.cpu()) for k,v in model.state_dict().items()):raise ValueError('Inference mutated frozen model state')
        report.update(model_state_invariance=True,calibration_inference_after_freeze=True,calibration_labels_accessed=False,selection=selection,completed_epochs=len(history))
    except Exception as e:
        import traceback;traceback.print_exc();report.update(status='failed',error_type=type(e).__name__,error=str(e))
    report['elapsed_seconds']=time.monotonic()-started
    return finish(out,report)
