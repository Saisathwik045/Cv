"""Training-only nested cohort construction and fixed loss-study decisions."""
import numpy as np
from trustretina.config import CLASSES
from trustretina.data.group_folds import allocate_components, ALGORITHM
from trustretina.data.validation import canonical_hash
from trustretina.evaluation.threshold_cv import support

ARMS = ('ratio_bce', 'sqrt_ratio_bce')
MATRIX = tuple((f'fold{k}_{a}', k, a) for k in range(3) for a in ARMS)


def validate_config(c):
    fixed={'schema_version':11,'backbone':'convnext_tiny','source_role':'train','outer_folds':3,
           'inner_folds':4,'inner_selection_fold':0,'split_seed':2026,'seed':2026,'image_size':224,
           'batch_size':16,'max_epochs':25,'learning_rate':1e-4,'minimum_learning_rate':1e-6,
           'weight_decay':.01,'early_stopping_patience':6,'early_stopping_min_delta':1e-4,
           'initialization':'torchvision_IMAGENET1K_V1','precision':'float32','optimizer':'AdamW',
           'schedule':'cosine','augmentation':'none','threshold':.5,
           'checkpoint_selection':'maximum_inner_patient_macro_auroc_tie_earliest',
           'patient_aggregation':'mean_left_right_sigmoid_before_threshold',
           'primary_comparison':'mean_paired_outer_macro_auroc_difference',
           'minimum_outer_positive_components':10,'minimum_inner_positive_components':10,
           'minimum_fit_positive_components':10,'minimum_negative_components':10,
           'maximum_gpu_total_seconds':14400,'maximum_gpu_session_seconds':2100,
           'gpu_overhead_reserve_per_attempt_seconds':300,'maximum_sessions':6,
           'maximum_attempts_per_run':2,'minimum_submission_interval_seconds':180,
           'bootstrap_replicates':2000,'bootstrap_seed':2026}
    if any(c.get(k)!=v for k,v in fixed.items()) or c['arms']!=list(ARMS) or c['class_order']!=list(CLASSES):
        raise ValueError('Prespecified study settings changed')
    if any(c.get(k) is not False for k in ('calibration_fitting','threshold_fitting','risk_policy_fitting',
        'historical_final_access','old_validation_access','full_training_refit')):
        raise ValueError('Disallowed data role or fitting')
    return c


def patient_rows(rows):
    patients={};eyes={}
    for i,row in enumerate(rows):
        pid=str(row['patient_id']);g=str(row['component_id']);y=row['labels'];eye=row['eye']
        if len(y)!=8 or not np.isin(y,[0,1]).all() or eye not in ('left','right'):
            raise ValueError('Binary labels and eye identity required')
        item={'patient_id':pid,'component_id':g,'labels':y}
        if pid in patients and patients[pid]!=item:raise ValueError('Patient labels/component conflict')
        if eye in eyes.setdefault(pid,{}):raise ValueError('Repeated eye')
        patients[pid]=item;eyes[pid][eye]=i
    if not patients or any(set(e)!= {'left','right'} for e in eyes.values()):raise ValueError('Both eyes required')
    return sorted(patients.values(),key=lambda x:x['patient_id'])


def allocate(patients,k):
    units=[[p['patient_id'] for p in patients if p['component_id']==g]
           for g in sorted({p['component_id'] for p in patients})]
    return allocate_components(patients,units,{'seed':2026,'split':{
        'fractions':[1/k]*k,'minimum_positive_patients':5,'minimum_negative_patients':10}})[0]


def role_summary(patients):
    return {'patients':len(patients),'images':2*len(patients),
            'components':len({p['component_id'] for p in patients}),
            'per_class':support([p['labels'] for p in patients],[p['component_id'] for p in patients])}


def audit_fold(rows,fold,cfg):
    all_indices=set(range(len(rows)));seen=set();patient_owner={};component_owner={};summary={}
    for role in ('fit','inner','outer'):
        indices=fold[role+'_indices']
        if len(set(indices))!=len(indices) or not set(indices)<=all_indices or seen&set(indices):
            raise ValueError('Missing/overlapping/invalid image indices')
        seen.update(indices);subset=patient_rows([rows[i] for i in indices])
        for p in subset:
            for key,owner in (('patient_id',patient_owner),('component_id',component_owner)):
                if p[key] in owner and owner[p[key]]!=role:raise ValueError('Patient/component role leakage')
                owner[p[key]]=role
        summary[role]=role_summary(subset)
        floor=cfg['minimum_'+role+'_positive_components']
        for c in summary[role]['per_class']:
            if c['positive_components']<floor or c['negative_components']<cfg['minimum_negative_components']:
                raise ValueError(f'Sparse {role} support for {c["label"]}; no seed search')
    if seen!=all_indices:raise ValueError('Missing training images')
    return summary


def build_manifest(rows,cfg,rows_sha256):
    validate_config(cfg);patients=patient_rows(rows);outer=allocate(patients,3);folds=[];outer_seen=set()
    for k in range(3):
        development=sorted([p for i,g in enumerate(outer) if i!=k for p in g],key=lambda p:p['patient_id'])
        inner=allocate(development,4)
        ids={'outer':{p['patient_id'] for p in outer[k]},'inner':{p['patient_id'] for p in inner[0]},
             'fit':{p['patient_id'] for g in inner[1:] for p in g}}
        fold={'outer_fold':k,**{r+'_indices':[i for i,row in enumerate(rows) if str(row['patient_id']) in values] for r,values in ids.items()}}
        fold['summary']=audit_fold(rows,fold,cfg);folds.append(fold)
        if outer_seen&ids['outer']:raise ValueError('Repeated outer assessment patient')
        outer_seen.update(ids['outer'])
    if outer_seen!={p['patient_id'] for p in patients}:raise ValueError('Incomplete outer coverage')
    result={'schema_version':11,'algorithm':ALGORITHM,'seed':2026,'source_role':'train',
            'train_rows_sha256':rows_sha256,'training':role_summary(patients),'folds':folds}
    result['sha256']=canonical_hash(result)
    return result


def audit_manifest(rows,manifest,cfg):
    if manifest.get('sha256')!=canonical_hash({k:v for k,v in manifest.items() if k!='sha256'}):raise ValueError('Fold manifest changed')
    if len(manifest['folds'])!=3 or manifest['source_role']!='train':raise ValueError('Wrong cohort/fold count')
    seen=set()
    for k,fold in enumerate(manifest['folds']):
        if fold['outer_fold']!=k or fold['summary']!=audit_fold(rows,fold,cfg):raise ValueError('Fold audit mismatch')
        components={rows[i]['component_id'] for i in fold['outer_indices']}
        if seen&components:raise ValueError('Repeated outer component')
        seen.update(components)
    if seen!={r['component_id'] for r in rows}:raise ValueError('Incomplete outer assessment coverage')


def loss_weights(rows,arm):
    if arm not in ARMS:raise ValueError('Unknown loss arm')
    patients=patient_rows(rows);y=np.asarray([p['labels'] for p in patients]);pos=y.sum(0);neg=len(y)-pos
    if (pos==0).any() or (neg==0).any():raise ValueError('Fitting outcome absent')
    ratio=neg/pos
    return ratio if arm=='ratio_bce' else np.sqrt(ratio)


def require_approval(plan,approval):
    if not approval or approval.get('user_authorized') is not True:raise PermissionError('Explicit v11 GPU budget approval required')
    for key in ('plan_sha256','experiment_source_sha256'):
        if approval.get(key)!=plan[key]:raise PermissionError('Approval does not bind exact plan/source')
    if approval.get('maximum_gpu_seconds')!=14400 or approval.get('run_ids')!=[r[0] for r in MATRIX] or not approval.get('approval_reference'):
        raise PermissionError('Approval must bind six runs and four GPU quota hours')


def reserve_attempt(plan,run):
    attempts=[a for r in plan['runs'] for a in r['attempts']]
    if len(attempts)>=6 or len(run['attempts'])>=2 or sum(a['reserved_seconds'] for a in attempts)+2400>14400:
        raise ValueError('Session/four-hour cap reached; preserve unfinished queue')
    return 2400


def outer_after_freeze(report,complete,callback):
    """Defer even opening outer pixels until checkpoint selection has ended."""
    if complete is not True or report.get('strict_best_reload') is not True:
        raise PermissionError('Outer assessment requires completed training and strict best reload')
    if report.get('outer_assessed'):raise ValueError('Outer assessment already performed; do not repeat')
    result=callback();report['outer_assessed']=True
    return result
