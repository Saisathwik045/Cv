"""Frozen research metrics, weighted estimands and paired component-cluster intervals. No fitting."""
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from trustretina.config import CLASSES
from .metrics import binary_targets,probabilities
from trustretina.reliability.development import component_weights


def _weights(n,values):
    w=np.ones(n) if values is None else np.asarray(values,dtype=float)
    if w.shape!=(n,) or not np.isfinite(w).all() or (w<=0).any():raise ValueError('Finite strictly positive row weights required')
    return w


def classification(y,p,weights=None):
    y=binary_targets(y);p=probabilities(p,y.shape);w=_weights(len(y),weights);pred=p>=.5
    tp=(w[:,None]*(y*pred)).sum(0);fp=(w[:,None]*((1-y)*pred)).sum(0);fn=(w[:,None]*(y*(~pred))).sum(0)
    divide=lambda a,b:np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)
    precision=divide(tp,tp+fp);recall=divide(tp,tp+fn);f1=divide(2*tp,2*tp+fp+fn);per={};auroc=[];ap=[]
    for j,c in enumerate(CLASSES):
        pos=int(y[:,j].sum());neg=len(y)-pos;both=pos>0 and neg>0
        roc=float(roc_auc_score(y[:,j],p[:,j],sample_weight=w)) if both else None
        avgp=float(average_precision_score(y[:,j],p[:,j],sample_weight=w)) if both else None
        auroc.append(roc);ap.append(avgp)
        per[c]={'positive':pos,'negative':neg,'predicted_positive':int(pred[:,j].sum()),'false_negative':int(((y[:,j]==1)&~pred[:,j]).sum()),'positive_weight':float((w*y[:,j]).sum()),'negative_weight':float((w*(1-y[:,j])).sum()),'predicted_positive_weight':float((w*pred[:,j]).sum()),'false_negative_weight':float(fn[j]),'auroc':roc,'auprc_average_precision':avgp,'precision':float(precision[j]),'recall':float(recall[j]),'f1':float(f1[j]),'false_negative_rate':float(fn[j]/(tp[j]+fn[j])) if pos else None,'precision_denominator_zero':bool(tp[j]+fp[j]==0),'recall_denominator_zero':bool(tp[j]+fn[j]==0),'ranking_undefined_reason':None if both else 'Both outcomes required'}
    micro_den=2*tp.sum()+fp.sum()+fn.sum();error=pred!=y;hamming=float(np.average(error.mean(1),weights=w));any_error=float(np.average(error.any(1),weights=w))
    return {'n':len(y),'weight_mass':float(w.sum()),'macro_auroc':float(np.mean(auroc)) if all(v is not None for v in auroc) else None,'macro_auprc_average_precision':float(np.mean(ap)) if all(v is not None for v in ap) else None,'macro_f1':float(f1.mean()),'micro_f1':float(2*tp.sum()/micro_den) if micro_den else 0.,'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),'micro_precision':float(tp.sum()/(tp.sum()+fp.sum())) if tp.sum()+fp.sum() else 0.,'micro_recall':float(tp.sum()/(tp.sum()+fn.sum())) if tp.sum()+fn.sum() else 0.,'subset_exact_match_accuracy':1-any_error,'labelwise_binary_accuracy':1-hamming,'hamming_loss':hamming,'any_label_error_rate':any_error,'per_class':per}


def calibration(y,p,weights=None):
    y=binary_targets(y);p=probabilities(p,y.shape);w=_weights(len(y),weights);safe=p.clip(1e-12,1-1e-12)
    losses=-(y*np.log(safe)+(1-y)*np.log1p(-safe));per={}
    for j,c in enumerate(CLASSES):
        bins=np.minimum((p[:,j]*10).astype(int),9);ece=0.;diagram=[]
        for b in range(10):
            mask=bins==b;n=int(mask.sum());mass=float(w[mask].sum())
            if n:
                meanp=float(np.average(p[mask,j],weights=w[mask]));prevalence=float(np.average(y[mask,j],weights=w[mask]));ece+=mass/w.sum()*abs(meanp-prevalence)
            else:meanp=prevalence=None
            diagram.append({'bin':b,'n':n,'weight_mass':mass,'mean_probability':meanp,'positive_fraction':prevalence})
        per[c]={'nll':float(np.average(losses[:,j],weights=w)),'brier':float(np.average((p[:,j]-y[:,j])**2,weights=w)),'ece':float(ece),'bins':diagram}
    return {'nll':float(np.average(losses.mean(1),weights=w)),'brier':float(np.average(((p-y)**2).mean(1),weights=w)),'macro_ece':float(np.mean([x['ece'] for x in per.values()])),'per_class':per}


def weighted_curve(y,p,score,weights=None):
    y=binary_targets(y);p=probabilities(p,y.shape);w=_weights(len(y),weights);score=np.asarray(score,dtype=float)
    if score.shape!=(len(y),) or not np.isfinite(score).all():raise ValueError('Finite aligned scores required')
    order=np.argsort(-score,kind='stable');s=score[order];w=w[order];error=(p[order]>=.5)!=y[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1];mass=w.cumsum()[ends];coverage=mass/w.sum();hamming=(w*error.mean(1)).cumsum()[ends]/mass;any_error=(w*error.any(1)).cumsum()[ends]/mass;delta=np.diff(np.r_[0.,coverage])
    return {'thresholds':s[ends],'accepted':ends+1,'coverage':coverage,'patient_coverage':(ends+1)/len(y),'hamming_risk':hamming,'any_error_risk':any_error,'hamming_aurc':float(delta@hamming),'any_error_aurc':float(delta@any_error)}


def frozen_grid(y,p,score,weights,cutoffs,*,detailed):
    y=np.asarray(y);p=np.asarray(p);score=np.asarray(score);w=_weights(len(y),weights);result=[]
    for old in cutoffs:
        target=float(old['target_coverage']);threshold=old['threshold'];mask=np.zeros(len(y),dtype=bool) if target==0 else score>=threshold
        n=int(mask.sum());mass=float(w[mask].sum());item={'nominal_validation_coverage':target,'frozen_validation_threshold':threshold,'accepted':n,'patient_coverage':n/len(y),'coverage':mass/float(w.sum()),'hamming_risk':None,'any_error_risk':None}
        if n:
            error=(p[mask]>=.5)!=y[mask];item.update(hamming_risk=float(np.average(error.mean(1),weights=w[mask])),any_error_risk=float(np.average(error.any(1),weights=w[mask])))
        if detailed:item['classification']=classification(y[mask],p[mask],w[mask]) if n else None
        result.append(item)
    return result


def scalar_values(bundle):
    out={}
    for key,v in bundle['classification'].items():
        if key not in ('n','weight_mass','per_class'):out['classification.'+key]=v
    for label,v in bundle['classification']['per_class'].items():
        for metric in ('auroc','auprc_average_precision','precision','recall','f1','false_negative_rate'):out['classification.'+label+'.'+metric]=v[metric]
    for metric in ('nll','brier','macro_ece'):out['calibration.'+metric]=bundle['calibration'][metric]
    for label,v in bundle['calibration']['per_class'].items():
        for metric in ('nll','brier','ece'):out['calibration.'+label+'.'+metric]=v[metric]
    return out


def statistics(y,predictions,scores,weightings,cutoffs,*,detailed=False):
    values={};result={'metrics':{},'reliability':{}};curves={}
    for mode,p in predictions.items():
        result['metrics'][mode]={}
        for weighting,w in weightings.items():
            bundle={'classification':classification(y,p,w),'calibration':calibration(y,p,w)};result['metrics'][mode][weighting]=bundle
            for key,value in scalar_values(bundle).items():values[mode+'.'+weighting+'.'+key]=value
    for weighting in weightings:
        raw=scalar_values(result['metrics']['raw'][weighting]);cal=scalar_values(result['metrics']['calibrated'][weighting])
        for key in raw:
            if key.startswith('calibration.'):values['paired_calibrated_minus_raw.'+weighting+'.'+key]=cal[key]-raw[key] if raw[key] is not None and cal[key] is not None else None
    for strategy,score in scores.items():
        mode=strategy.split('__')[0];p=predictions[mode];result['reliability'][strategy]={}
        for weighting,w in weightings.items():
            curve=weighted_curve(y,p,score,w);grid=frozen_grid(y,p,score,w,cutoffs[strategy],detailed=detailed)
            result['reliability'][strategy][weighting]={'hamming_aurc':curve['hamming_aurc'],'any_error_aurc':curve['any_error_aurc'],'saved_validation_grid':grid};curves[strategy+'__'+weighting]=curve
            for metric in ('hamming_aurc','any_error_aurc'):values['reliability.'+strategy+'.'+weighting+'.'+metric]=curve[metric]
            for i,item in enumerate(grid):
                for metric in ('coverage','patient_coverage','hamming_risk','any_error_risk'):values['grid.'+strategy+'.'+weighting+'.'+str(i)+'.'+metric]=item[metric]
    for mode in predictions:
        for weighting in weightings:
            for addition in ('confidence_uncertainty','confidence_uncertainty_quality'):
                for metric in ('hamming_aurc','any_error_aurc'):
                    values['paired_vs_confidence.'+mode+'__'+addition+'.'+weighting+'.'+metric]=result['reliability'][mode+'__'+addition][weighting][metric]-result['reliability'][mode+'__confidence'][weighting][metric]
    for name in ('confidence','confidence_uncertainty','confidence_uncertainty_quality'):
        for weighting in weightings:
            for metric in ('hamming_aurc','any_error_aurc'):values['paired_calibrated_minus_raw.reliability.'+name+'.'+weighting+'.'+metric]=result['reliability']['calibrated__'+name][weighting][metric]-result['reliability']['raw__'+name][weighting][metric]
    return values,result,curves


def bootstrap(y,predictions,scores,components,cutoffs,*,replicates=200,seed=62026,minimum_defined=160,progress_callback=None):
    y=binary_targets(y);components=np.asarray(components);groups=sorted(set(components));mapping={g:np.flatnonzero(components==g) for g in groups};rng=np.random.default_rng(seed)
    baseweights={'patient_equal':np.ones(len(y)),'component_equal':component_weights(components)};samples={}
    for b in range(replicates):
        idx=np.concatenate([mapping[groups[j]] for j in rng.integers(0,len(groups),len(groups))])
        # Reuse original 1/group-size on each occurrence; repeated draws retain mass.
        values,_,_=statistics(y[idx],{k:v[idx] for k,v in predictions.items()},{k:v[idx] for k,v in scores.items()},{k:v[idx] for k,v in baseweights.items()},cutoffs)
        for key,value in values.items():samples.setdefault(key,[]).append(np.nan if value is None else value)
        if progress_callback and (b+1)%25==0:progress_callback(b+1,replicates)
    intervals={}
    for key,values in samples.items():
        v=np.asarray(values,dtype=float);defined=v[np.isfinite(v)];enough=len(defined)>=minimum_defined
        intervals[key]={'lower':float(np.quantile(defined,.025)) if enough else None,'upper':float(np.quantile(defined,.975)) if enough else None,'defined_replicates':len(defined),'total_replicates':replicates,'confidence':.95,'method':'paired whole-component percentile bootstrap, fixed candidate; not simultaneous clinical guarantees'}
    return intervals,{key:np.asarray(v) for key,v in samples.items()}
