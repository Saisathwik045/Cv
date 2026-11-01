#!/usr/bin/env python3
"""Finalize aggregate quota and frozen-artifact integrity evidence; no inference."""
import datetime,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import controlled_experiment as c


def record():
    registry=json.loads((c.BASE/'registry.json').read_text())
    c.verify_plan(registry)
    if any(r['status']!='complete' for r in registry['runs']):raise ValueError('Complete matrix required for final provenance')
    quota=json.loads((c.BASE/'latest-quota.json').read_text());baseline=registry['quota_baseline']
    if quota['refresh_time']!=baseline['refresh_time']:raise ValueError('Quota counter crossed refresh; retain raw evidence and resolve accounting')
    if not quota['paid_scaling_disabled']:raise ValueError('Paid scaling must remain disabled')
    observed=quota['used_seconds']-baseline['used_seconds']
    reserved=sum(a['reserved_seconds'] for r in registry['runs'] for a in r['attempts'])
    if not 0<=observed<=14400 or not 0<=reserved<=14400:raise ValueError('Four-hour cap exceeded or quota counter inconsistent')
    before=json.loads((c.BASE/'protected-before.json').read_text());after={}
    for path in before:
        h=hashlib.sha256()
        with (ROOT/path).open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        after[path]=h.hexdigest()
    c.write(c.BASE/'protected-after.json',after)
    if before!=after:raise ValueError('A protected frozen artifact changed')
    artifacts=[]
    for run in registry['runs']:
        for attempt in run['attempts']:
            folder=ROOT/attempt['bundle']
            verification=json.loads((folder/'verification.json').read_text())
            if not verification.get('integrity_verified'):raise ValueError('Unverified attempt in completed matrix')
            report=json.loads((folder/'download'/verification['report_relative_path']).read_text())
            artifacts.append({'run':run['id'],'kernel_version':verification['kernel_version'],'status':report['status'],'source_sha256':verification['source_sha256'],'report_sha256':verification['artifact_sha256'],'output_sha256':report['output_hashes']})
    output={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':registry['plan_sha256'],'source_commit':registry['source_commit'],'experiment_source_sha256':registry['experiment_source_sha256'],'observed_quota_seconds_including_resumes':observed,'conservative_reserved_seconds':reserved,'maximum_seconds':14400,'paid_scaling_disabled':True,'quota_observed_utc':quota['observed_utc'],'quota_refresh_time':quota['refresh_time'],'completed_runs':len(registry['runs']),'submitted_sessions':sum(len(r['attempts']) for r in registry['runs']),'artifacts':artifacts,'worker_seconds':sum(r['total_worker_seconds'] for r in registry['runs']),'protected_artifacts_unchanged':True,'protected_sha256':after,'protection_check':'Opaque-byte SHA256 only; no calibration/final arrays parsed','new_demo_model_installed':False,'training_driver_continues_after_task':False}
    c.write(ROOT/'results/round2-v9/execution-provenance.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='protected_sha256'},indent=2))


if __name__=='__main__':record()
