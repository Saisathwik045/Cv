"""Paired outer-fold comparisons; no model, threshold or calibration fitting."""
import numpy as np
from trustretina.training.comparison import comparison_metrics
from trustretina.evaluation.threshold_cv import summarize,support


def metrics(y,p,groups):
    result=comparison_metrics(y,p)
    counts=summarize(y,np.asarray(p)>=.5)
    result['confusion']=counts['per_class'];result['component_support']=support(y,groups)
    return result


def contrast(a,b):
    result={}
    for key in ('macro_auroc','macro_auprc_average_precision','macro_f1'):
        result[key]=b[key]-a[key] if a[key] is not None and b[key] is not None else None
    for key in ('recall','f1'):
        result['H_'+key]=b['per_class']['H'][key]-a['per_class']['H'][key] if a['per_class']['H']['positive'] and b['per_class']['H']['positive'] else None
    return result


def paired_summary(folds):
    if len(folds)!=3:raise ValueError('Exactly three completed paired outer folds required')
    seen_patients=set();seen_groups=set();reports=[];pooled={a:[] for a in ('ratio_bce','sqrt_ratio_bce')}
    for k,pair in enumerate(folds):
        a,b=(pair[n] for n in pooled)
        for key in ('patient_ids','component_ids','labels'):
            if not np.array_equal(a[key],b[key]):raise ValueError('Paired patient/group/label mismatch')
        patients=set(a['patient_ids']);groups=set(a['component_ids'])
        if len(patients)!=len(a['labels']) or patients&seen_patients or groups&seen_groups:raise ValueError('Outer assessment duplication/leakage')
        seen_patients|=patients;seen_groups|=groups
        m={arm:metrics(data['labels'],data['probabilities'],data['component_ids']) for arm,data in pair.items()}
        reports.append({'outer_fold':k,'arms':m,'paired_differences':contrast(m['ratio_bce'],m['sqrt_ratio_bce'])})
        for arm in pooled:pooled[arm].append(pair[arm])
    means={arm:{key:float(np.mean([r['arms'][arm][key] for r in reports])) if all(r['arms'][arm][key] is not None for r in reports) else None for key in ('macro_auroc','macro_auprc_average_precision','macro_f1','micro_f1')} for arm in pooled}
    if any(m['macro_auroc'] is None for m in means.values()):raise ValueError('All eight ranking outcomes required; no partial winner')
    candidate='sqrt_ratio_bce' if means['sqrt_ratio_bce']['macro_auroc']>means['ratio_bce']['macro_auroc'] else 'ratio_bce'
    secondary={arm:metrics(np.concatenate([d['labels'] for d in ds]),np.concatenate([d['probabilities'] for d in ds]),np.concatenate([d['component_ids'] for d in ds])) for arm,ds in pooled.items()}
    return {'folds':reports,'mean_fold_metrics':means,'candidate_for_future_study':candidate,
            'mean_paired_macro_auroc_difference':means['sqrt_ratio_bce']['macro_auroc']-means['ratio_bce']['macro_auroc'],
            'secondary_pooled':secondary,'patients':len(seen_patients),'components':len(seen_groups),
            'warning':'Exploratory nested development; pooled scores come from different models; no independent confirmation or deployment.'}


def bootstrap_replicate(folds,index,seed=2026):
    """Reproducible independently addressable replicate: supports checkpointed CPU batches."""
    rng=np.random.default_rng(np.random.SeedSequence([seed,index]));differences=[]
    for pair in folds:
        reference=pair['ratio_bce'];groups=np.asarray(reference['component_ids']);unique=sorted(set(groups))
        if len(unique)<2:raise ValueError('At least two assessment components required')
        samples=rng.integers(0,len(unique),len(unique));indices=np.concatenate([np.flatnonzero(groups==unique[i]) for i in samples])
        arm_metrics=[comparison_metrics(pair[arm]['labels'][indices],pair[arm]['probabilities'][indices]) for arm in ('ratio_bce','sqrt_ratio_bce')]
        differences.append(contrast(*arm_metrics))
    return {key:float(np.mean([d[key] for d in differences])) if all(d[key] is not None for d in differences) else None for key in differences[0]}


def bootstrap_intervals(replicates):
    if not replicates:raise ValueError('Bootstrap replicates required')
    result={}
    for key in replicates[0]:
        values=[r[key] for r in replicates if r[key] is not None]
        interval=np.quantile(values,[.025,.975],method='linear').tolist() if len(values)>=.9*len(replicates) else None
        result[key]={'percentile_95_interval':interval,'defined_replicates':len(values),'undefined_replicates':len(replicates)-len(values),'total_replicates':len(replicates),'minimum_defined_fraction_for_interval':.9}
    return result
