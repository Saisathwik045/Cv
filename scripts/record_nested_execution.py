#!/usr/bin/env python3
"""Finalize API-counter accounting and frozen-artifact preservation; no GPU calls."""
import datetime,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import nested_experiment as c
from drive_nested_queue import seconds


def main():
    p=json.loads((c.BASE/'registry.json').read_text());c.verify_plan(p)
    if any(r['status'] in ('running','submitting','ambiguous') for r in p['runs']):raise ValueError('Resolve active/ambiguous jobs before final accounting')
    q=json.loads((c.BASE/'final-quota.json').read_text());gpu=q['raw']['gpuQuota']
    if q['refresh_time']!=p['quota_baseline']['refresh_time']:raise ValueError('Quota window changed; accounting review required')
    charged=q['used_seconds']-p['quota_baseline']['used_seconds']
    reserved=sum(a['reserved_seconds'] for r in p['runs'] for a in r['attempts'])
    if not 0<=charged<=14400 or reserved>14400:raise ValueError('Four-hour quota/reservation limit exceeded')
    if not q['paid_scaling_disabled'] or seconds(gpu['timeReserved'])!=0:raise ValueError('GPU quota must be settled with paid scaling disabled')
    attempts=[a for r in p['runs'] for a in r['attempts']]
    versions=[a['kernel_version'] for a in attempts if a.get('kernel_version')]
    if len(attempts)>6 or len(versions)!=len(set(versions)):raise ValueError('Session cap or duplicate remote version')
    audit=json.loads((ROOT/'results/nested-v11/execution-audit.json').read_text())
    if audit['plan_sha256']!=p['plan_sha256']:raise ValueError('Wrong verified artifact audit')
    expected=json.loads((ROOT/'results/nested-v11/readiness.json').read_text())['original_demo_hashes_unchanged']
    actual={path:c.file_hash(ROOT/path) for path in expected}
    if expected!=actual:raise ValueError('Original inference bundle/contract changed')
    changed=subprocess.check_output(['git','diff','3b9856bed08b56f34ab4c4100d8fce515da9214c','--name-only'],cwd=ROOT,text=True).splitlines()
    if any(path.startswith(('results/final-v7/','results/reliability-v6/','src/trustretina/demo/','configs/development-v4.toml')) for path in changed):raise ValueError('Protected historical/application paths changed')
    rejections=[entry for run in p['runs'] for entry in run.get('pre_session_save_rejections',[])]
    worker=sum(a.get('elapsed_seconds',0) for a in attempts)
    output={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':p['plan_sha256'],
        'approved_worker_source_sha256':p['experiment_source_sha256'],'implementation_source_commit':p['source_commit'],
        'gpu_quota_baseline_used_seconds':p['quota_baseline']['used_seconds'],'gpu_quota_final_used_seconds':q['used_seconds'],
        'actual_charged_gpu_quota_seconds':charged,'actual_charged_gpu_quota_hours':charged/3600,
        'worker_elapsed_seconds_separate_from_charged_quota':worker,'conservative_reserved_seconds':reserved,
        'maximum_charged_seconds':14400,'maximum_sessions':6,'reserved_scientific_attempts':len(attempts),
        'actual_versioned_gpu_sessions':len(versions),'versions':versions,'paid_scaling_disabled':True,
        'final_gpu_reserved_seconds':0,'quota_observed_utc':q['observed_utc'],'quota_refresh_time':q['refresh_time'],
        'complete_matrix':all(r['status']=='complete' for r in p['runs']),'pre_session_rejected_save_requests':rejections,
        'pre_session_request_note':'SaveKernel400 produced no version, no owned job and no extra charged quota; existing slot/source/nonce retained. Request/recovery evidence preserved; no budget refund or seventh GPU session.',
        'verified_artifacts':audit['artifact_inventory'],'inner_selection_audits':audit['inner_checkpoint_checks'],
        'original_demo_hashes_unchanged':actual,'protected_tracked_files_unchanged':True,
        'historical_final_payload_opened':False,'calibration_dataset_opened':False,'threshold_fitted':False,
        'old_calibration_temperature_applied':False,'full_data_refit':False,'demo_replaced':False,
        'review_required_unchanged':True,'monitoring_continues_after_task':False}
    c.write(ROOT/'results/nested-v11/execution-provenance.json',output)
    print(json.dumps({k:output[k] for k in ('complete_matrix','actual_versioned_gpu_sessions','actual_charged_gpu_quota_seconds','worker_elapsed_seconds_separate_from_charged_quota','conservative_reserved_seconds','demo_replaced')},indent=2))


if __name__=='__main__':main()
