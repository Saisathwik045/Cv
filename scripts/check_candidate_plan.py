#!/usr/bin/env python3
"""Validate a non-executable v14 proposal on CPU; no SDK, data acquisition or fitting."""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def validate(plan,datasets):
    if plan['status']!='planning_only_not_execution_authorized' or plan['execution_ready'] is not False:
        raise ValueError('This proposal cannot authorize execution')
    if any(plan['authorization'][k] for k in ('submit_gpu_jobs','fit_models','fit_calibration','fit_thresholds','publish_or_replace_demo')):
        raise ValueError('Research-only authorization boundary changed')
    original=json.loads((ROOT/'results/reliability-v6/candidate-contract.json').read_text())
    if plan['class_order']!=list('NDGCAHMO') or plan['frozen_v4_release_sha256']!=original['release_sha256']:
        raise ValueError('Frozen class order/release changed')
    if plan['partitions'].keys()!={'train','validation','calibration'}:raise ValueError('Forbidden/missing development role')
    for role,n in [('train',2275),('validation',525),('calibration',350)]:
        p=plan['partitions'][role]
        if p['patients']!=n or p['manifest_sha256']!=original['role_manifest_sha256'][role]:raise ValueError('Frozen role identity changed')
        if set(p['positive_patients'])!=set('NDGCAHMO') or any(p['positive_patients'][k]+p['negative_patients'][k]!=n for k in 'NDGCAHMO'):
            raise ValueError('Wrong class support accounting')
    a,b=plan['runs']
    if [a['id'],b['id']]!=['v14_candidate_res320_seed2026','v14_reference_res224_seed2026'] or [a['image_size'],b['image_size']]!=[320,224] or a['settings']!=b['settings']:
        raise ValueError('Expected two matched fits differing only in resolution/time reservation')
    settings=a['settings']
    fixed={'backbone':'convnext_tiny','loss':'ratio_bce','seed':2026,'max_epochs':25,'learning_rate':1e-4,'minimum_learning_rate':1e-6,'weight_decay':.01,'optimizer':'AdamW','precision':'float32','dropout':.2,'schedule':'cosine_25_epoch_horizon','micro_batch_size':8,'accumulation_steps':2,'early_stopping_patience':6,'early_stopping_min_delta':1e-4,'augmentation':'none'}
    if any(settings.get(k)!=v for k,v in fixed.items()):raise ValueError('Prespecified candidate training settings changed')
    if (settings['fit_role'],settings['selection_role'],settings['calibration_role'])!=('train','validation','calibration'):
        raise ValueError('Fitting/selection role leakage')
    if settings['patient_aggregation']!='mean_left_right_sigmoid_before_threshold' or settings['decision_threshold']!=.5:
        raise ValueError('Mean-eye predictor/threshold changed')
    if any(settings[k] for k in ('historical_final_access','threshold_fitting','risk_policy_fitting','demo_replacement')):
        raise ValueError('Forbidden fitting/data access')
    if plan['historical_final']['use']!='excluded from training,selection,fitting,new confirmation and sample-case selection':
        raise ValueError('Historical final cohort must stay excluded')
    calibration=plan['calibration']
    if calibration['old_temperature_reuse'] or calibration['old_bundle_mutation'] or calibration['execute_now'] or calibration['fit_role']!='calibration':
        raise ValueError('Old calibration or role boundary changed')
    evaluation=plan['evaluation']
    if evaluation['unknown_labels']!='masked, never negative' or not evaluation['all_eight_macro_requires_all_eight_compatible_labels'] or evaluation['external_full_eight_ready']:
        raise ValueError('Unknown labels/full-eight readiness misrepresented')
    if datasets['all_eight_independent_cohort_identified'] or any(d['full_eight_compatible_verified'] or d['files_accessed'] for d in datasets['options']):
        raise ValueError('Documentation-only review cannot invent full-eight access/compatibility')
    compute=plan['compute'];reserved=sum(x['reservation_seconds'] for x in plan['runs'])
    if not compute['free_kaggle_only'] or not compute['paid_scaling_must_be_disabled'] or compute['maximum_gpu_quota_seconds']!=14400 or compute['maximum_sessions_including_failures_and_resumes']!=3 or compute['budget_approval_obtained']:
        raise ValueError('Proposed free-only unapproved cap/session boundary changed')
    if any(x['reservation_seconds']!=x['worker_timeout_seconds']+x['overhead_reservation_seconds'] for x in plan['runs']):
        raise ValueError('Session overhead missing from reservation')
    if reserved!=12000 or reserved+compute['optional_single_resume_max_worker_seconds']+compute['optional_resume_overhead_seconds']!=14400:
        raise ValueError('Reservations exceed/misstate full cap including optional resume')
    if not plan['blocking_prerequisites'] or any(plan['prepared_inputs'][k] is not None for k in ('validation320_sha256','calibration320_sha256','new_worker_sha256')):
        raise ValueError('Unfinished prerequisites must not be represented as verified inputs/worker')
    return {'valid_reviewable_proposal':True,'execution_ready':False,'GPU_jobs_authorized':0,'fitting_performed':False,
            'proposed_gpu_cap_seconds':14400,'proposed_session_cap':3,'blocking_prerequisites':plan['blocking_prerequisites']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,default=ROOT/'results/candidate-v14/proposed-plan.json');args=parser.parse_args()
    datasets=json.loads((ROOT/'results/candidate-v14/dataset-options.json').read_text())
    print(json.dumps(validate(json.loads(args.plan.read_text()),datasets),indent=2))
