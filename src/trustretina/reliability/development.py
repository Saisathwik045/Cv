"""Prospective Phase 6 signals and descriptive development curves. No clinical risk fit."""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score
from trustretina.evaluation.metrics import binary_targets, probabilities
from .quality import quality_proxies


def canonical_quality(chw):
    x=np.asarray(chw)
    if x.ndim!=3 or x.shape[0]!=3 or x.dtype!=np.uint8:raise ValueError('Require canonical uint8 CHW')
    x=x.transpose(1,2,0);fill=np.array([124,116,104],dtype=np.uint8)
    same=np.all(x==fill,axis=2);rows=np.flatnonzero(~same.all(1));cols=np.flatnonzero(~same.all(0))
    if not len(rows) or not len(cols):return np.zeros(3)
    roi=x[rows[0]:rows[-1]+1,cols[0]:cols[-1]+1]
    if min(roi.shape[:2])<2:return np.zeros(3)
    q=quality_proxies(roi)
    return np.asarray([q[k] for k in ('foreground_fraction','contrast','gradient_energy')])


def quality_midrank(training,values):
    t=np.asarray(training,dtype=float);v=np.asarray(values,dtype=float)
    if t.ndim!=2 or t.shape[1]!=3 or v.ndim!=2 or v.shape[1]!=3 or not len(t) or not np.isfinite(t).all() or not np.isfinite(v).all():raise ValueError('Require finite training and evaluation quality matrices')
    return np.column_stack([(np.searchsorted(np.sort(t[:,j]),v[:,j],side='left')+np.searchsorted(np.sort(t[:,j]),v[:,j],side='right'))/(2*len(t)) for j in range(3)]).mean(1)


def patient_quality(rows,eye_scores,patient_ids):
    q={}
    if len(rows)!=len(eye_scores):raise ValueError('Quality alignment mismatch')
    for row,score in zip(rows,eye_scores):
        item=q.setdefault(row['patient_id'],{})
        if row['eye'] in item:raise ValueError('Repeated quality eye')
        item[row['eye']]=float(score)
    if set(q)!=set(patient_ids) or any(set(x)!={'left','right'} for x in q.values()):raise ValueError('Both eyes required for quality')
    return np.asarray([min(q[pid].values()) for pid in patient_ids])


def temperature_probabilities(p,temperature):
    p=probabilities(p)
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('Positive shared temperature required')
    p=p.clip(1e-6,1-1e-6);z=(np.log(p)-np.log1p(-p))/temperature
    return np.exp(-np.logaddexp(0.,-z))


def component_weights(component_ids):
    c=np.asarray(component_ids)
    if c.ndim!=1 or not len(c):raise ValueError('Component IDs required')
    _,inverse,counts=np.unique(c,return_inverse=True,return_counts=True)
    return 1/counts[inverse]


def nll(y,p,weights=None):
    y=binary_targets(y);p=probabilities(p,y.shape).clip(1e-12,1-1e-12)
    row=-(y*np.log(p)+(1-y)*np.log1p(-p)).mean(1)
    return float(np.average(row,weights=weights))


def fit_shared_temperature(p,y,component_ids,*,role,allow_fitting=False):
    if role!='calibration' or not allow_fitting:raise ValueError('Only explicitly authorized calibration fitting is permitted')
    y=binary_targets(y);p=probabilities(p,y.shape)
    if len(component_ids)!=len(y) or len(set(component_ids))<50 or (y.sum(0)==0).any() or (y.sum(0)==len(y)).any():raise ValueError('Shared calibration needs >=50 components and all outcomes')
    weights=component_weights(component_ids)
    objective=lambda log_t:nll(y,temperature_probabilities(p,np.exp(log_t)),weights)
    result=minimize_scalar(objective,bounds=(np.log(.05),np.log(20.)),method='bounded',options={'xatol':1e-8,'maxiter':500})
    t=float(np.exp(result.x));identity=objective(0.);loss=float(result.fun)
    # Identity is a feasible numerical reference, not held-out optimization.
    if identity<loss:t=1.;loss=identity
    boundary=t<=.05*(1+1e-5) or t>=20*(1-1e-5)
    if not result.success or not np.isfinite(loss) or boundary:raise ValueError('Temperature optimizer failed or reached numeric boundary; candidate freeze blocked')
    return {'temperature':t,'objective':'component-equal eight-label patient NLL after mean-eye probabilities','fitted_component_weighted_nll':loss,'identity_component_weighted_nll':identity,'optimizer_success':bool(result.success),'optimizer_evaluations':int(result.nfev),'boundary_hit':bool(boundary),'role':'calibration','interpretation':'in-sample fit diagnostic; no held-out improvement claim'}


def signals(p,patient_passes,quality):
    p=probabilities(p);samples=np.asarray(patient_passes,dtype=float);q=np.asarray(quality,dtype=float)
    if samples.ndim!=3 or samples.shape[1:]!=p.shape or samples.shape[0]<2 or not np.isfinite(samples).all() or samples.min()<0 or samples.max()>1 or q.shape!=(len(p),) or not np.isfinite(q).all() or q.min()<0 or q.max()>1:raise ValueError('Misaligned reliability signals')
    confidence=np.maximum(p,1-p).min(1);u=samples.var(0,ddof=0).max(1)
    cu=confidence*np.clip(1-4*u,0,1)
    return {'confidence':confidence,'confidence_uncertainty':cu,'confidence_uncertainty_quality':cu*q,'uncertainty':u,'quality':q}


def coverage_curve(y,p,score):
    y=binary_targets(y);p=probabilities(p,y.shape);score=np.asarray(score,dtype=float)
    if score.shape!=(len(y),) or not np.isfinite(score).all():raise ValueError('Misaligned reliability score')
    order=np.argsort(-score,kind='stable');s=score[order];errors=(p[order]>=.5)!=y[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    sizes=ends+1;coverage=sizes/len(y)
    hamming=errors.mean(1).cumsum()[ends]/sizes;any_error=errors.any(1).cumsum()[ends]/sizes
    increments=np.diff(np.r_[0.,coverage])
    return {'thresholds':s[ends],'accepted':sizes,'coverage':coverage,'hamming_risk':hamming,'any_error_risk':any_error,'hamming_aurc':float(increments@hamming),'any_error_aurc':float(increments@any_error)}


def fixed_coverage_summary(y,p,score,grid=(0.,.1,.2,.3,.5,.75,1.)):
    curve=coverage_curve(y,p,score);result=[];y=np.asarray(y);p=np.asarray(p)
    for target in grid:
        if target==0:
            result.append({'target_coverage':0.,'coverage':0.,'accepted':0,'threshold':None,'hamming_risk':None,'any_error_risk':None,'per_class':None});continue
        i=min(int(np.searchsorted(curve['coverage'],target,side='left')),len(curve['coverage'])-1)
        threshold=float(curve['thresholds'][i]);accepted=np.asarray(score)>=threshold;a=y[accepted];pred=p[accepted]>=.5
        support=[]
        for j in range(8):
            positive=int(a[:,j].sum());fn=int(((a[:,j]==1)&~pred[:,j]).sum())
            support.append({'positive':positive,'negative':len(a)-positive,'false_negative':fn,'false_negative_rate':fn/positive if positive else None,'sensitivity':1-fn/positive if positive else None,'f1':float(f1_score(a[:,j],pred[:,j],zero_division=0))})
        result.append({'target_coverage':target,'coverage':float(accepted.mean()),'accepted':int(accepted.sum()),'threshold':threshold,'hamming_risk':float((pred!=a).mean()),'any_error_risk':float((pred!=a).any(1).mean()),'macro_f1':float(f1_score(a,pred,average='macro',zero_division=0)),'per_class':support})
    return result


def calibration_diagnostics(y,p,component_ids):
    y=binary_targets(y);p=probabilities(p,y.shape);weights=component_weights(component_ids);per_class=[]
    for j in range(8):
        bins=np.minimum((p[:,j]*10).astype(int),9);ece=0.;diagram=[]
        for k in range(10):
            mask=bins==k;n=int(mask.sum())
            if n:
                probability=float(p[mask,j].mean());positive=float(y[mask,j].mean());ece+=n/len(y)*abs(probability-positive)
            else:probability=positive=None
            diagram.append({'bin':k,'n':n,'mean_probability':probability,'positive_fraction':positive})
        per_class.append({'positive':int(y[:,j].sum()),'negative':int(len(y)-y[:,j].sum()),'brier':float(((p[:,j]-y[:,j])**2).mean()),'ece':float(ece),'bins':diagram})
    return {'patient_nll':nll(y,p),'component_equal_nll':nll(y,p,weights),'brier':float(((p-y)**2).mean()),'component_equal_brier':float(np.average(((p-y)**2).mean(1),weights=weights)),'macro_ece_10_equal_width_bins':float(np.mean([v['ece'] for v in per_class])),'per_class':per_class}
