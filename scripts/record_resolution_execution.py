#!/usr/bin/env python3
"""Closeout actual charged quota and verified artifacts; no API calls or training."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import resolution_experiment as c


def main():
    p=json.loads((c.BASE/'registry.json').read_text());c.verify_plan(p)
    q=json.loads((c.BASE/'final-quota.json').read_text());audit=json.loads((ROOT/'results/resolution-v13/execution-audit.json').read_text())
    if q['refresh_time']!=p['quota_baseline']['refresh_time']:raise ValueError('Quota window changed; reconcile accounting')
    if not q['paid_disabled'] or q['reserved_seconds']!=0:raise ValueError('Final accounting is not settled or paid scaling enabled')
    charged=q['used_seconds']-p['quota_baseline']['used_seconds'];attempts=[a for r in p['runs'] for a in r['attempts']]
    versions=[a['kernel_version'] for a in attempts if a.get('kernel_version')];reserved=sum(a['reserved_seconds'] for a in attempts)
    if not 0<=charged<=21600 or reserved>21600 or len(attempts)>6 or len(set(versions))!=len(versions):raise ValueError('Approved cap/session/unique-version invariant failed')
    if any(r['status'] in ('running','submitting','ambiguous') for r in p['runs']):raise ValueError('Resolve active/ambiguous provider state first')
    if audit['plan_sha256']!=p['plan_sha256']:raise ValueError('Wrong artifact audit')
    inventory=json.loads((c.BASE/'final-remote-inventory.json').read_text())
    if set(inventory['matching_v13_training_refs'])!={v.rsplit('/',1)[0] for v in versions}:
        raise ValueError('Owned remote training inventory differs from recorded attempts')
    observed=[]
    for path in (c.BASE/'accounting').glob('quota-*.json'):
        raw=json.loads(path.read_text())['raw']
        if raw.get('quotaRefreshTime')==p['quota_baseline']['refresh_time']:observed.append(c.seconds(raw['gpuQuota']['timeUsed'])-p['quota_baseline']['used_seconds'])
    peak=max([charged,*observed])
    if peak>21600:raise ValueError('An observed charge exceeded cap; preserve evidence')
    changed=subprocess.check_output(['git','diff','35ccc5adb497dd64f4650b0c2a1e46a4b879493f','--name-only'],cwd=ROOT,text=True).splitlines()
    protected=('src/','configs/resolution-v13.toml','docs/RESOLUTION_V13.md','scripts/resolution_runtime.py','scripts/resolution_study.py','scripts/resolution_experiment.py','scripts/report_resolution_results.py','results/final-v7/','results/reliability-v6/','results/threshold-v10/','streamlit_app.py')
    if any(any(path==key or key.endswith('/') and path.startswith(key) for key in protected) for path in changed):raise ValueError('Frozen scientific/historical/application paths changed')
    result={'plan_sha256':p['plan_sha256'],'worker_source_sha256':p['experiment_source_sha256'],'config_sha256':p['config_sha256'],
            'complete_matrix':all(r['status']=='complete' for r in p['runs']),'actual_charged_free_GPU_quota_seconds':charged,'actual_charged_free_GPU_quota_hours':charged/3600,
            'peak_observed_charged_GPU_quota_seconds':peak,'worker_elapsed_seconds_separate_from_quota':sum(a.get('elapsed_seconds',0) for a in attempts),
            'quota_baseline_used_seconds':p['quota_baseline']['used_seconds'],'final_quota_used_seconds':q['used_seconds'],'quota_refresh':q['refresh_time'],
            'maximum_quota_seconds':21600,'maximum_sessions':6,'reserved_budget_seconds':reserved,'reserved_attempts':len(attempts),'actual_versioned_sessions':len(versions),
            'versions':versions,'owned_remote_inventory_reconciled':True,'paid_scaling_disabled':True,'final_GPU_reserved_seconds':0,'artifacts':audit['artifacts'],
            'original_private_demo_bundle_loaded_this_execution':False,'original_demo_replaced':False,'historical_final_payload_opened':False,
            'calibration_payload_opened':False,'calibration_fitted':False,'threshold_fitted':False,'full_data_refit':False,'frozen_tracked_paths_unchanged':True,
            'patient_artifacts_private':True,'approval_record_sha256':c.file_hash(c.BASE/'approval.json'),'no_monitoring_after_closeout':True}
    c.write(ROOT/'results/resolution-v13/execution-provenance.json',result)
    print(json.dumps({k:result[k] for k in ('complete_matrix','actual_charged_free_GPU_quota_seconds','actual_versioned_sessions','worker_elapsed_seconds_separate_from_quota')}))


if __name__=='__main__':main()
