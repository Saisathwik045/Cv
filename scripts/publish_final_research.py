#!/usr/bin/env python3
"""Generate all permitted held-out aggregate tables/report from verified private artifacts."""
import csv,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trustretina.data.validation import file_hash
from trustretina.runner.verified_artifacts import read_stage


def number(value,digits=4):return 'undefined' if value is None else f'{value:.{digits}f}'


def publish():
    base=ROOT/'artifacts/runner/phase7';r=json.loads((base/'registry.json').read_text())
    if set(r['stages'])!={'inputs','inference','analysis'} or any(v['status']!='complete' for v in r['stages'].values()):raise ValueError('All three verified final stages required')
    report,folder,verified=read_stage(ROOT/r['stages']['analysis']['bundle'])
    if report['status']!='complete' or report['candidate_contract_sha256']!=r['candidate_contract_sha256'] or report['parameter_fitting'] is not False:raise ValueError('Final analysis provenance mismatch')
    summary=json.loads((folder/'aggregate-summary.json').read_text());dest=ROOT/'results/final-v7';dest.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/'aggregate-summary.json',dest/'aggregate-summary.json')
    for weighting in ('patient_equal','component_equal'):
        for name in ('hamming_risk','any_error_risk','reliability-diagrams'):shutil.copyfile(folder/(weighting+'-'+name+'.png'),dest/(weighting+'-'+name+'.png'))
    estimates=summary['estimates'];intervals=summary['intervals'];metrics=summary['results']['metrics'];reliability=summary['results']['reliability'];order=summary['class_order'];rows=[]
    for key,value in estimates.items():
        interval=intervals[key];rows.append({'endpoint':key,'estimate':value,'lower_95':interval['lower'],'upper_95':interval['upper'],'defined_replicates':interval['defined_replicates'],'total_replicates':interval['total_replicates']})
    def csv_write(name,data,fields=None):
        with (dest/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields or list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
    csv_write('endpoints-and-intervals.csv',rows)
    per_class=[]
    for mode in ('raw','calibrated'):
        for weighting in ('patient_equal','component_equal'):
            for label in order:
                value=metrics[mode][weighting]['classification']['per_class'][label];row={'predictor':mode,'weighting':weighting,'class':label,**value}
                for metric in ('auroc','auprc_average_precision','precision','recall','f1','false_negative_rate'):
                    interval=intervals[mode+'.'+weighting+'.classification.'+label+'.'+metric]
                    row[metric+'_lower_95']=interval['lower'];row[metric+'_upper_95']=interval['upper'];row[metric+'_defined_replicates']=interval['defined_replicates']
                per_class.append(row)
    csv_write('per-class-metrics.csv',per_class)
    grids=[]
    for strategy,by_weight in reliability.items():
        for weighting,item in by_weight.items():
            for grid in item['saved_validation_grid']:
                for label in order:
                    value=grid['classification']['per_class'][label] if grid['classification'] else {}
                    grids.append({'strategy':strategy,'weighting':weighting,'nominal_validation_coverage':grid['nominal_validation_coverage'],'actual_final_coverage':grid['coverage'],'actual_final_patient_coverage':grid['patient_coverage'],'retrospective_subset_patients':grid['accepted'],'class':label,'positive':value.get('positive',0),'negative':value.get('negative',0),'predicted_positive':value.get('predicted_positive',0),'false_negative':value.get('false_negative',0),'positive_weight':value.get('positive_weight',0.),'negative_weight':value.get('negative_weight',0.),'false_negative_weight':value.get('false_negative_weight',0.),'false_negative_rate':value.get('false_negative_rate'),'precision':value.get('precision'),'recall':value.get('recall'),'f1':value.get('f1'),'hamming_risk':grid['hamming_risk'],'any_error_risk':grid['any_error_risk']})
    csv_write('saved-grid-support-and-errors.csv',grids)
    def ci(key,digits=4):
        value=estimates[key];interval=intervals[key]
        if value is None:return 'undefined'
        if interval['lower'] is None:return number(value,digits)+' [interval undefined; '+str(interval['defined_replicates'])+'/'+str(interval['total_replicates'])+']'
        return number(value,digits)+' ['+number(interval['lower'],digits)+', '+number(interval['upper'],digits)+']'
    lines=['# Final held-out research evaluation','',
           'Generated from checksum-verified prediction and evaluation artifacts under the protocol committed before final exposure. No training, refitting, threshold tuning, strategy selection or final reporting-case selection occurred.','',
           '**Primary endpoint — raw patient-equal macro AUROC: '+ci('raw.patient_equal.classification.macro_auroc')+'.**',
           f"All **{summary['patients']} patients / {summary['images']} eyes / {summary['components']} frozen containment components** were evaluated. Brackets denote fixed-seed 95% component-cluster percentile intervals, conditional on the frozen candidate. Secondary intervals are not multiplicity-adjusted clinical guarantees.",'',
           '## Patient-level class results','',
           'Raw probabilities; fixed p >= 0.5. Positive shared temperature preserves every binary decision, so calibrated precision/recall/F1/accuracy are identical. Both raw/calibrated ranking metrics and both weighting schemes are in the CSV/JSON, including all undefined flags and replicate counts.','',
           '| Class | Positive / negative | AUROC [95%] | AUPRC/AP [95%] | Precision | Recall | F1 [95%] |','|---|---:|---|---|---:|---:|---|']
    for label in order:
        value=metrics['raw']['patient_equal']['classification']['per_class'][label];prefix='raw.patient_equal.classification.'+label+'.'
        lines.append('| '+label+f" | {value['positive']} / {value['negative']} | "+ci(prefix+'auroc')+' | '+ci(prefix+'auprc_average_precision')+' | '+number(value['precision'])+' | '+number(value['recall'])+' | '+ci(prefix+'f1')+' |')
    lines+=['','## Overall classification','',
            '| Predictor / weighting | Macro AUROC [95%] | Macro F1 [95%] | Micro F1 [95%] | Exact-match accuracy | Labelwise accuracy | Hamming loss |','|---|---|---|---|---:|---:|---:|']
    for mode in ('raw','calibrated'):
        for weighting in ('patient_equal','component_equal'):
            prefix=mode+'.'+weighting+'.classification.';m=metrics[mode][weighting]['classification'];lines.append('| '+mode+' / '+weighting+' | '+ci(prefix+'macro_auroc')+' | '+ci(prefix+'macro_f1')+' | '+ci(prefix+'micro_f1')+' | '+number(m['subset_exact_match_accuracy'])+' | '+number(m['labelwise_binary_accuracy'])+' | '+number(m['hamming_loss'])+' |')
    lines+=['','Macro F1 averages eight class F1s. Micro F1 uses weighted TP/FP/FN on the (n,8) arrays. Exact match requires all eight labels correct; labelwise accuracy averages correct binary labels; Hamming loss is its complement. Any-label error is one minus exact-match accuracy. Precision/recall/F1 zero denominators use zero with flags; ranking requires both outcomes and macro AUROC requires all eight classes.','',
            '## Frozen calibration on held-out patients','',
            f"The reused shared temperature is **T={summary['temperature_reused']:.9f}**. It acts on logit(clipped mean eye probabilities), not mean eye logits. No parameter or normalizer was refitted.",'',
            '| Predictor / weighting | NLL [95%] | Brier [95%] | Macro ECE [95%] |','|---|---|---|---|']
    for mode in ('raw','calibrated'):
        for weighting in ('patient_equal','component_equal'):
            prefix=mode+'.'+weighting+'.calibration.';lines.append('| '+mode+' / '+weighting+' | '+ci(prefix+'nll')+' | '+ci(prefix+'brier')+' | '+ci(prefix+'macro_ece')+' |')
    lines+=['','Paired differences are **calibrated minus raw**, on identical component resamples. Negative values favor calibration for these losses.','',
            '| Weighting | Δ NLL [95%] | Δ Brier [95%] | Δ macro ECE [95%] |','|---|---|---|---|']
    for weighting in ('patient_equal','component_equal'):
        prefix='paired_calibrated_minus_raw.'+weighting+'.calibration.';lines.append('| '+weighting+' | '+ci(prefix+'nll',6)+' | '+ci(prefix+'brier',6)+' | '+ci(prefix+'macro_ece',6)+' |')
    lines+=['','Classwise paired calibration differences, patient-equal (calibrated minus raw):','',
            '| Class | Δ NLL [95%] | Δ Brier [95%] | Δ ECE [95%] |','|---|---|---|---|']
    for label in order:
        prefix='paired_calibrated_minus_raw.patient_equal.calibration.'+label+'.'
        lines.append('| '+label+' | '+ci(prefix+'nll',6)+' | '+ci(prefix+'brier',6)+' | '+ci(prefix+'ece',6)+' |')
    lines+=['','ECE uses ten fixed equal-width probability bins per class, weighted bin gaps and an equal mean across eight classes. It is sensitive to binning and sparse classes. [Patient reliability diagrams](patient_equal-reliability-diagrams.png) · [Component-equal diagrams](component_equal-reliability-diagrams.png).','',
            '## All fixed reliability ablations','',
            'Lower AURC is better. Full curves use every complete tied-score prefix and right-step integration. These are retrospective research ranking analyses, **not a validated automatic acceptance policy**. No useful-looking strategy or cutoff is selected.','',
            '| Fixed strategy | Weighting | Hamming AURC [95%] | Any-label-error AURC [95%] | Δ Hamming AURC vs confidence [95%] |','|---|---|---|---|---|']
    strategy_order=[mode+'__'+name for mode in ('raw','calibrated') for name in ('confidence','confidence_uncertainty','confidence_uncertainty_quality')]
    for strategy in strategy_order:
        for weighting in ('patient_equal','component_equal'):
            prefix='reliability.'+strategy+'.'+weighting+'.';diff='reference' if strategy.endswith('__confidence') else ci('paired_vs_confidence.'+strategy+'.'+weighting+'.hamming_aurc',6)
            lines.append('| '+strategy+' | '+weighting+' | '+ci(prefix+'hamming_aurc',6)+' | '+ci(prefix+'any_error_aurc',6)+' | '+diff+' |')
    lines+=['','### Actual patient coverage at the saved development cutoffs','',
            'Nominal targets are validation coverage. The same saved score thresholds are applied unchanged; actual final coverage can differ. No final quantile is selected as an operating point. At nominal zero, no patients are retained and risk is undefined. The all-patient endpoint is separately reported above.','',
            '| Validation target | Raw C | Raw C+U | Raw C+U+Q | Cal C | Cal C+U | Cal C+U+Q |','|---:|---:|---:|---:|---:|---:|---:|']
    for i,target in enumerate(r['config']['coverage_grid']):lines.append('| '+number(target,2)+' | '+' | '.join(number(reliability[key]['patient_equal']['saved_validation_grid'][i]['coverage'],3) for key in strategy_order)+' |')
    lines+=['','[All-eight retained support, false-negative counts/rates, precision/recall/F1 and actual weighted coverage](saved-grid-support-and-errors.csv). FNR/sensitivity are undefined with no positive support; empty-subset risk is undefined. Hamming risk can conceal negative-label imbalance; any-label-error curves are reported alongside it.','',
            '[Patient Hamming curves](patient_equal-hamming_risk.png) · [Patient any-label-error curves](patient_equal-any_error_risk.png) · [Component Hamming curves](component_equal-hamming_risk.png) · [Component any-label-error curves](component_equal-any_error_risk.png).','',
            '## Contribution and limitations','',
            'The study demonstrates a reproducible, frozen multilabel research pipeline with auditable patient/group separation, calibration, trained-head variability, technical quality and explanation interfaces. It evaluates one selected checkpoint; it cannot establish superiority over untested final-cohort alternatives.']
    def direction(key):
        interval=intervals[key]
        if interval['lower'] is None:return 'has an undefined interval because support is insufficient'
        if interval['upper']<0:return 'is lower with a paired interval below zero on this cohort'
        if interval['lower']>0:return 'is higher with a paired interval above zero on this cohort'
        return 'has a paired interval including zero; a clear improvement is not established'
    lines+=['', 'Calibrated-minus-raw patient NLL '+direction('paired_calibrated_minus_raw.patient_equal.calibration.nll')+'.']
    class_groups={'lower_interval':[],'higher_interval':[],'interval_includes_zero_or_undefined':[]}
    for label in order:
        interval=intervals['paired_calibrated_minus_raw.patient_equal.calibration.'+label+'.nll']
        key='interval_includes_zero_or_undefined' if interval['lower'] is None or interval['lower']<=0<=interval['upper'] else ('higher_interval' if interval['lower']>0 else 'lower_interval')
        class_groups[key].append(label)
    lines+=['', 'Pooled calibration gains were heterogeneous. Classwise NLL paired intervals were below zero for '+', '.join(class_groups['lower_interval'])+'; above zero (worse) for '+', '.join(class_groups['higher_interval'])+'; and included zero or were undefined for '+', '.join(class_groups['interval_includes_zero_or_undefined'])+'. These are prespecified secondary, non-simultaneous intervals; no class-specific recalibration was performed.']
    for mode in ('raw','calibrated'):
        for name,label in [('confidence_uncertainty','Adding head uncertainty'),('confidence_uncertainty_quality','Adding technical quality to head uncertainty')]:
            key='paired_vs_confidence.'+mode+'__'+name+'.patient_equal.hamming_aurc';lines.append(label+' ('+mode+') versus confidence-only: Hamming AURC '+direction(key)+'.')
    lines+=['','These secondary comparisons do not authorize changing the frozen pipeline. Clinical automatic triage, complete epistemic uncertainty, clinical quality grading and lesion localization are not established. Deployment remains **REVIEW REQUIRED for all '+str(summary['patients'])+' patients**; no ACCEPT threshold is fitted or introduced.','']
    lines += ['- '+text for text in summary['limitations']]
    lines+=['','Bootstrap intervals use '+str(r['config']['bootstrap_replicates'])+' resamples, seed '+str(r['config']['bootstrap_seed'])+', of whole frozen components. Original inverse-component-size weights are preserved on each repeated draw. Intervals condition on the fixed model/temperature and have limited tail precision. Valid/total replicate counts are explicit; fewer than '+str(r['config']['minimum_defined_bootstrap_replicates'])+' defined values yields an undefined interval.','',
            '## Provenance and next phase','',
            'The protocol/configuration was committed before final exposure. Frozen contract, selected checkpoint, temperature, normalizer, development cutoffs, complete final coverage, model-state invariance, sources/nonces and every retrieved artifact checksum passed verification. Private predictions, images/features, score thresholds and bootstrap arrays stay outside Git in pinned private Kaggle outputs. No failed/replaced attempt or refit is hidden.','',
            '[Full permitted aggregate results](aggregate-summary.json) · [All endpoints and intervals](endpoints-and-intervals.csv) · [Per-class metrics](per-class-metrics.csv) · [Run registry](run-registry.json) · [Protocol](../../docs/FINAL_EVALUATION_V7.md).','',
            'Next phase: package this same frozen inference pipeline into the demo app, retaining its research limitations and REVIEW REQUIRED default. Do not retrain, recalibrate or tune against these final results.','']
    (dest/'REPORT.md').write_text('\n'.join(lines));print(json.dumps({'primary_estimate':summary['primary_estimate'],'primary_interval':summary['primary_interval'],'published':'permitted aggregate report/tables/figures only'}))

if __name__=='__main__':publish()
