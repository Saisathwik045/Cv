"""Prospective second-round decisions; no final/calibration data or automatic fit."""
import math
import numpy as np
from trustretina.evaluation.metrics import binary_targets,probabilities

MATRIX=[('convnext_constant','convnext_tiny','constant'),('convnext_cosine','convnext_tiny','cosine'),('densenet_constant','densenet121','constant'),('densenet_cosine','densenet121','cosine')]


def validate_config(cfg):
    actual=[(r['id'],r['backbone'],r['schedule']) for r in cfg['runs']]
    if actual!=MATRIX or cfg['schema_version']!=9:raise ValueError('Fixed four-run matrix changed')
    required={'max_epochs':25,'seed':2026,'image_size':224,'augmentation':'none','loss':'BCEWithLogitsLoss','positive_weights':'negative_over_positive_training_patients_only_unclipped','learning_rate':1e-4,'minimum_learning_rate':1e-6,'weight_decay':.01,'batch_size':16,'early_stopping_patience':6,'early_stopping_min_delta':1e-4,'maximum_gpu_session_seconds':1800,'maximum_gpu_total_seconds':14400,'maximum_run_reserved_seconds':4800,'gpu_overhead_reserve_per_attempt_seconds':600,'optimizer':'AdamW','precision':'float32','num_workers':0,'maximum_attempts_per_run':2,'initialization':'torchvision_IMAGENET1K_V1','patient_aggregation':'mean_left_right_sigmoid_before_threshold','threshold':.5,'checkpoint_selection':'maximum_validation_patient_macro_auroc_all_eight_defined_tie_earliest'}
    if any(cfg.get(k)!=v for k,v in required.items()):raise ValueError('Prespecified training/budget rule changed')
    if any(cfg.get(k) is not False for k in ('final_evaluation_access','calibration_fitting','risk_policy_fitting','threshold_fitting','higher_resolution_enabled','alternative_loss_enabled')):raise ValueError('Disallowed role/factor/fitting enabled')
    if cfg.get('class_order')!=['N','D','G','C','A','H','M','O'] or cfg.get('backbones')!=['convnext_tiny','densenet121']:raise ValueError('Backbone/class order changed')
    return cfg


def advance_auc(state,score,epoch,*,min_delta=1e-4,patience=6):
    if score is None or not math.isfinite(score):raise ValueError('All-eight AUROC must be defined')
    state=dict(state);improved=score>state.get('best_score',-math.inf)
    if improved:state.update(best_score=float(score),best_epoch=epoch)
    if score>state.get('early_stop_reference',-math.inf)+min_delta:state.update(early_stop_reference=float(score),bad_epochs=0)
    else:state['bad_epochs']=state.get('bad_epochs',0)+1
    state['early_stopped']=state['bad_epochs']>=patience
    return state,improved


def require_approval(plan,approval):
    if not approval or approval.get('user_authorized') is not True:raise PermissionError('New explicit user approval required; no submission authorized')
    for key in ('plan_sha256','experiment_source_sha256'):
        if approval.get(key)!=plan[key]:raise PermissionError('Approval does not bind this exact plan/source')
    if approval.get('maximum_gpu_seconds')!=14400 or approval.get('run_ids')!=[x[0] for x in MATRIX] or not approval.get('approval_reference'):raise PermissionError('Approval must bind four exact runs and four-hour cap')


def reserve_attempt(plan,run):
    seconds=plan['config']['maximum_gpu_session_seconds']+plan['config']['gpu_overhead_reserve_per_attempt_seconds']
    if len(run['attempts'])>=2:raise ValueError('Two-attempt run limit reached; preserve progress and seek revised budget')
    if sum(a['reserved_seconds'] for r in plan['runs'] for a in r['attempts'])+seconds>14400:raise ValueError('Four-hour reservation cap reached')
    if sum(a['reserved_seconds'] for a in run['attempts'])+seconds>4800:raise ValueError('Per-run reservation cap reached')
    return seconds


def training_oof_thresholds(y,p,*,role,positive_components,negative_components,allow_fitting=False):
    """Future separate threshold study ONLY; no CLI invokes it in this queue.

    Maximize per-class F1 on training-only OOF outputs. Independent maxima maximize
    macro F1. Fixed .05:.05:.95 grid; ties nearest .5, then higher threshold.
    Sparse (<10 positive/negative supplying components) labels retain .5.
    These fitted-on-OOF diagnostics still require independent evaluation.
    """
    if not allow_fitting or role!='train_oof':raise PermissionError('Separate training-only OOF threshold experiment required')
    y=binary_targets(y);p=probabilities(p,y.shape);result=[]
    if len(positive_components)!=8 or len(negative_components)!=8:raise ValueError('All-eight component support required')
    for j in range(8):
        if positive_components[j]<10 or negative_components[j]<10:result.append(.5);continue
        values=[]
        for tick in range(1,20):
            t=tick/20;pred=p[:,j]>=t;tp=np.sum(pred&(y[:,j]==1));fp=np.sum(pred&(y[:,j]==0));fn=np.sum(~pred&(y[:,j]==1));den=2*tp+fp+fn
            values.append((2*tp/den if den else 0,-abs(tick-10),t))
        result.append(max(values)[2])
    return np.asarray(result)
