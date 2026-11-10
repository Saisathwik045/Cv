"""CPU-testable v14 role, predictor, approval and quota invariants. No job submission."""
import math,time
import numpy as np
MATRIX=(('v14_candidate_res320_seed2026',320),('v14_reference_res224_seed2026',224))


def validate_config(c):
    fixed={'expected_visible_cuda_devices':2,'active_cuda_device':0,'backbone':'convnext_tiny','loss':'ratio_bce','seed':2026,'max_epochs':25,'learning_rate':1e-4,'minimum_learning_rate':1e-6,'weight_decay':.01,'optimizer':'AdamW','schedule':'cosine_25_epoch_horizon','precision':'float32','micro_batch_size':8,'accumulation_steps':2,'effective_batch_size':16,'dropout':.2,'early_stopping_patience':6,'early_stopping_min_delta':1e-4,'augmentation':'none','fit_role':'train','selection_role':'validation','calibration_role':'calibration','patient_aggregation':'mean_left_right_sigmoid_before_threshold','decision_threshold':.5,'checkpoint_selection':'maximum_validation_patient_macro_auroc_tie_earliest','historical_final_access':False,'threshold_fitting':False,'risk_policy_fitting':False,'demo_replacement':False}
    if any(c.get(k)!=v for k,v in fixed.items()) or c['class_order']!=list('NDGCAHMO'):raise ValueError('Frozen v14 training configuration changed')
    return c


def audit_roles(roles,expected_counts=None):
    if set(roles)!={'train','validation','calibration'}:raise ValueError('Only frozen development roles permitted')
    patients=set();components=set();summary={}
    for name,rows in roles.items():
        paired={}
        for row in rows:
            if ('labels' in row)!=(name!='calibration'):raise ValueError('Calibration labels must not be mounted; fitting roles require labels')
            if name!='calibration' and (len(row['labels'])!=8 or not set(row['labels'])<={0,1}):raise ValueError('Expected eight binary patient targets')
            paired.setdefault(row['patient_id'],[]).append(row)
        groups=set()
        for pair in paired.values():
            if len(pair)!=2 or {r['eye'] for r in pair}!={'left','right'} or len({r['component_id'] for r in pair})!=1:raise ValueError('Both eyes must be a single complete patient/group')
            if name!='calibration' and pair[0]['labels']!=pair[1]['labels']:raise ValueError('Both eye targets must inherit the same original patient vector')
            groups.add(pair[0]['component_id'])
        if set(paired)&patients or groups&components:raise ValueError('Patient/containment group crosses roles')
        if expected_counts and len(paired)!=expected_counts[name]:raise ValueError('Frozen role patient count changed')
        patients|=set(paired);components|=groups;summary[name]={'patients':len(paired),'images':len(rows),'components':len(groups)}
    return summary


def train_weights(rows):
    y={}
    for r in rows:
        if r['patient_id'] in y and y[r['patient_id']]!=r['labels']:raise ValueError('Conflicting inherited targets')
        y[r['patient_id']]=r['labels']
    a=np.asarray(list(y.values()),dtype=np.int64)
    if a.ndim!=2 or a.shape[1]!=8 or not np.isin(a,[0,1]).all():raise ValueError('Eight binary TRAIN targets required')
    pos=a.sum(0)
    if (pos==0).any() or (pos==len(a)).any():raise ValueError('TRAIN class weights need both outcomes')
    return (len(a)-pos)/pos


def mean_eyes(rows,probabilities):
    p=np.asarray(probabilities,dtype=np.float64)
    if p.shape!=(len(rows),8) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Finite independent sigmoid probabilities required')
    groups={}
    for i,r in enumerate(rows):groups.setdefault(r['patient_id'],[]).append(i)
    ids=sorted(groups);values=[];components=[]
    for pid in ids:
        ix=groups[pid]
        if len(ix)!=2 or {rows[i]['eye'] for i in ix}!={'left','right'} or len({rows[i]['component_id'] for i in ix})!=1:raise ValueError('Cannot manufacture a paired predictor')
        values.append(p[ix].mean(0));components.append(rows[ix[0]]['component_id'])
    return {'patient_ids':np.asarray(ids),'component_ids':np.asarray(components),'probabilities':np.stack(values)}


def calibration_after_freeze(report,callback):
    if report.get('status')!='complete' or not report.get('strict_best_reload') or not report.get('selected_checkpoint_sha256'):
        raise ValueError('Calibration inference requires completed fit, strict reload and persisted checkpoint freeze')
    return callback()


def require_approval(p,a):
    required={'approved':True,'plan_sha256':p['plan_sha256'],'worker_source_sha256':p['worker_source_sha256'],'config_sha256':p['config_sha256'],'inputs_sha256':p['inputs_sha256'],'maximum_gpu_quota_seconds':14400,'maximum_sessions':3,'run_ids':[x[0] for x in MATRIX],'free_kaggle_only':True,'external_confirmation_not_ready_acknowledged':True}
    if not a or any(a.get(k)!=v for k,v in required.items()) or not a.get('approved_at_utc'):raise PermissionError('Separate exact v14 GPU execution approval required')


def next_budget(p,q,run,now=None):
    now=time.time() if now is None else now
    if not q.get('paid_disabled') or q.get('reserved_seconds')!=0 or not 0<=now-q.get('observed_unix',0)<=60:raise ValueError('Fresh settled free-only quota required')
    attempts=[a for r in p['runs'] for a in r['attempts']]
    if len(attempts)>=3 or len(run['attempts'])>=2:raise ValueError('Three-session/one-resume cap reached')
    if not attempts and q['free_seconds']<14400:raise ValueError('Full four-hour free quota required before first fit')
    baseline=p.get('quota_baseline',{'used_seconds':q['used_seconds'],'refresh_time':q['refresh_time']})
    charged=q['used_seconds']-baseline['used_seconds']
    if charged<0 or q['refresh_time']!=baseline['refresh_time']:raise ValueError('Quota counter/window changed; reconcile')
    if run['attempts']:
        if run['status']!='paused' or not run['attempts'][-1].get('resume_verified'):raise ValueError('Only verified compatible epoch checkpoints can resume')
        timeout=2100
    else:timeout=7500 if run['image_size']==320 else 3900
    rate=1
    if attempts:
        last=attempts[-1]
        if last.get('elapsed_seconds'):
            rate=max(1,math.ceil(max(0,q['used_seconds']-last['quota_used_before']-300)/last['elapsed_seconds']))
    reserve=timeout*rate+300
    if run['attempts'] and reserve>2400:raise ValueError('Resume accounting exceeds its remaining reservation')
    spent=sum(a['reserved_seconds'] for a in attempts)
    if max(charged,spent)+reserve>14400 or q['free_seconds']<reserve:raise ValueError('Actual charging/free quota cannot cover next bounded session')
    return timeout,reserve
