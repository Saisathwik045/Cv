"""Prospective v13 settings and exact accumulation; no job submission."""
import math
import numpy as np

MATRIX=tuple((f'fold{k}_res{size}',k,size) for k in range(3) for size in (224,320))


def validate_config(c):
    fixed={'schema_version':13,'backbone':'convnext_tiny','source_role':'train','seed':2026,
           'micro_batch_size':8,'accumulation_steps':2,'effective_batch_size':16,'max_epochs':25,
           'learning_rate':1e-4,'minimum_learning_rate':1e-6,'weight_decay':.01,
           'early_stopping_patience':6,'early_stopping_min_delta':1e-4,'optimizer':'AdamW',
           'precision':'float32','schedule':'cosine','augmentation':'none','loss':'ratio_bce',
           'initialization':'torchvision_IMAGENET1K_V1','threshold':.5,
           'patient_aggregation':'mean_left_right_sigmoid_before_threshold',
           'checkpoint_selection':'maximum_inner_patient_macro_auroc_tie_earliest',
           'primary_comparison':'mean_paired_outer_macro_auroc_difference',
           'minimum_outer_positive_components':10,'minimum_inner_positive_components':10,
           'minimum_fit_positive_components':10,'minimum_negative_components':10,
           'maximum_gpu_total_seconds':21600,'maximum_sessions':6,'overhead_seconds':300,
           'timeout_224_seconds':2400,'timeout_320_seconds':4200,'minimum_submission_interval_seconds':180,
           'bootstrap_replicates':2000,'bootstrap_seed':2026}
    if any(c.get(k)!=v for k,v in fixed.items()) or c.get('resolutions')!=[224,320] or c.get('class_order')!=list('NDGCAHMO'):
        raise ValueError('Prespecified resolution study settings changed')
    for k in ('calibration_fitting','threshold_fitting','historical_final_access','old_validation_access','risk_policy_fitting','full_training_refit'):
        if c.get(k) is not False:raise ValueError('Disallowed data access/fitting')
    return c


def require_approval(plan,approval):
    if not approval or approval.get('user_authorized') is not True:raise PermissionError('New exact v13 training approval required')
    for key in ('plan_sha256','experiment_source_sha256'):
        if approval.get(key)!=plan[key]:raise PermissionError('Approval does not bind frozen v13 identity')
    if approval.get('maximum_gpu_seconds')!=21600 or approval.get('maximum_sessions')!=6 or approval.get('run_ids')!=[r[0] for r in MATRIX] or not approval.get('approval_reference'):
        raise PermissionError('Approval must bind exact six runs and six free-quota hours')


def train_accumulated_epoch(model,loader,optimizer,criterion,device,*,accumulation_steps=2):
    """Mean over actual samples per optimizer window, including partial last window."""
    import torch
    if accumulation_steps<1 or loader.drop_last or not loader.batch_size:
        raise ValueError('Explicit non-dropping fixed microbatches required')
    model.train();seen=0;total=0.;steps=0;window_samples=0
    for index,(x,y) in enumerate(loader):
        if index%accumulation_steps==0:
            optimizer.zero_grad(set_to_none=True)
            window_samples=min(loader.batch_size*accumulation_steps,len(loader.dataset)-seen)
        x=x.to(device);y=y.to(device);loss=criterion(model(x),y)
        if loss.ndim!=0 or not torch.isfinite(loss):raise ValueError('Finite scalar mean loss required')
        (loss*(len(y)/window_samples)).backward()
        total+=float(loss.detach())*len(y);seen+=len(y)
        if (index+1)%accumulation_steps==0 or index+1==len(loader):
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Non-finite/missing gradients')
            optimizer.step();steps+=1
    if seen!=len(loader.dataset) or steps!=math.ceil(seen/(loader.batch_size*accumulation_steps)):
        raise ValueError('Optimizer cadence/sample coverage mismatch')
    return total/seen,steps
