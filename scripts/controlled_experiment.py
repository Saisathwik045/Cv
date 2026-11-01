#!/usr/bin/env python3
"""Bounded v9 queue. Preparation is authorized; submission requires a separate bound approval."""
import argparse,base64,contextlib,datetime,fcntl,io,json,re,subprocess,sys,tomllib,uuid,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import full_experiment as transport
from trustretina.data.validation import file_hash,canonical_hash
from trustretina.runner.verified_artifacts import read_stage
from trustretina.training.controlled import validate_config,require_approval,reserve_attempt
BASE=ROOT/'artifacts/runner/round2-v9';CONFIG=ROOT/'configs/controlled-training-v9.toml';DOC=ROOT/'docs/CONTROLLED_TRAINING_V9.md'
write=transport.write


def prepare():
    if (BASE/'registry.json').exists():raise ValueError('Existing queue must be resumed, not recreated')
    cfg=validate_config(tomllib.loads(CONFIG.read_text()))
    tracked=['configs/controlled-training-v9.toml','docs/CONTROLLED_TRAINING_V9.md','scripts/controlled_experiment.py']+[str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py')]
    subprocess.run(['git','ls-files','--error-unmatch',*tracked],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    subprocess.run(['git','diff','--exit-code','HEAD','--','configs/controlled-training-v9.toml','docs/CONTROLLED_TRAINING_V9.md','src','scripts/controlled_experiment.py'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    cache,folder,v=read_stage(ROOT/'artifacts/runner/full-v5-inputs');index=json.loads((folder/'inputs.json').read_text())
    if cache['status']!='complete' or set(index['roles'])!={'train','validation'} or index['release_sha256']!=cfg['release_manifest_sha256']:raise ValueError('Original checked train/validation cache required')
    runtime=json.loads((ROOT/'results/full-v5/runtime.json').read_text())
    identity={'config':cfg,'config_sha256':file_hash(CONFIG),'protocol_sha256':file_hash(DOC),'experiment_source_sha256':transport.source_hash(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'cache':{'kernel_version':v['kernel_version'],'inputs_sha256':file_hash(folder/'inputs.json'),'parent_config_sha256':index['config_sha256'],'report_sha256':v['artifact_sha256'],'bundle':'artifacts/runner/full-v5-inputs'},'expected_dependencies':runtime['dependencies'],'docker_image':runtime['docker_images']['convnext_tiny'],'development_audit_sha256':file_hash(ROOT/'results/round2-v9/development-audit.json')}
    plan={**identity,'plan_sha256':canonical_hash(identity),'runs':[{**r,'status':'queued','attempts':[]} for r in cfg['runs']],'execution_authorized':False,'queue_blocker':'Awaiting explicit user approval of four runs and four GPU hours','calibration_accessed':False,'historical_final_reused_for_selection':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(BASE/'registry.json',plan);export(plan);return plan


def export(plan):
    # Contains only configuration, aggregate state and receipt identities; no rows.
    receipts={a['bundle']:json.loads((ROOT/a['bundle']/'submission.json').read_text()) for r in plan['runs'] for a in r['attempts']}
    write(ROOT/'results/round2-v9/queue.json',{'registry':plan,'receipts':receipts})
    (BASE/'PRIVATE_INDEX.md').write_text('Registry is authoritative. No submission before separate user approval.\nReuse full-v5-inputs; never rebuild splits/cache. Each attempt keeps source, nonce,\nsubmission receipt, remote version, verification, checkpoints/history/predictions.\nAfter approval: advance --submit --approval-file <bound JSON>; one bounded status\ncheck per invocation. Reconcile ambiguous jobs; never blindly retry.\nHard reservations count all attempts including failures: four hours globally,\ntwo half-hour attempts per run, each charged 600 seconds overhead guard. No automatic extra matrix or paid fallback.\n')


def restore():
    if (BASE/'registry.json').exists():raise ValueError('Private registry exists; refusing overwrite')
    saved=json.loads((ROOT/'results/round2-v9/queue.json').read_text())
    for relative,receipt in saved['receipts'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(BASE.resolve()):raise ValueError('Receipt outside v9 private storage')
        path.mkdir(parents=True,exist_ok=True)
        if (path/'submission.json').exists() and json.loads((path/'submission.json').read_text())!=receipt:raise ValueError('Conflicting receipt; recover manually')
        write(path/'submission.json',receipt)
    write(BASE/'registry.json',saved['registry']);return saved['registry']


def verify_plan(plan):
    keys=('config','config_sha256','protocol_sha256','experiment_source_sha256','source_commit','cache','expected_dependencies','docker_image','development_audit_sha256')
    if canonical_hash({k:plan[k] for k in keys})!=plan['plan_sha256']:raise ValueError('Immutable plan identity changed')
    if plan['config']!=tomllib.loads(CONFIG.read_text()) or [(r['id'],r['backbone'],r['schedule']) for r in plan['runs']]!=[(r['id'],r['backbone'],r['schedule']) for r in plan['config']['runs']]:raise ValueError('Queue differs from fixed configuration')
    if file_hash(ROOT/'results/round2-v9/development-audit.json')!=plan['development_audit_sha256']:raise ValueError('Development audit changed')
    if file_hash(CONFIG)!=plan['config_sha256'] or file_hash(DOC)!=plan['protocol_sha256'] or transport.source_hash()!=plan['experiment_source_sha256']:raise ValueError('Plan/source changed; compatibility review and new approval required')
    validate_config(plan['config'])


def mine(api):
    found=[]
    for page in range(1,21):
        batch=list(api.kernels_list(mine=True,page=page,page_size=100,sort_by='dateRun'));found.extend(k.ref for k in batch)
        if len(batch)<100:return found
    raise ValueError('Remote inventory exceeds bounded pagination; reconcile manually')


def quota(api):
    raw=json.loads(str(api.quota_view()));q=raw['gpuQuota']
    def seconds(value):
        # Preserve raw response. Existing Kaggle SDK sometimes emits 2092.830.0s.
        match=re.fullmatch(r'(\d+(?:\.\d+)?)(?:\.0)?s',str(value))
        if not match:raise ValueError('Unrecognized quota duration; fail closed')
        return float(match.group(1))
    left=seconds(q['totalTimeAllowed'])-seconds(q['timeUsed'])-seconds(q['timeReserved'])
    record={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'raw':raw,'free_remaining_seconds':left,'used_seconds':seconds(q['timeUsed']),'refresh_time':raw.get('quotaRefreshTime'),'paid_scaling_disabled':q.get('isPayToScaleEnabled') is False}
    write(BASE/'latest-quota.json',record);return record


def build(plan,run,approval):
    require_approval(plan,approval);verify_plan(plan)
    reserved=reserve_attempt(plan,run);nonce=uuid.uuid4().hex
    bundle=BASE/run['id']/f'attempt-{len(run["attempts"])+1}';bundle.mkdir(parents=True,exist_ok=False)
    spec={'plan_sha256':plan['plan_sha256'],'protocol_sha256':plan['protocol_sha256'],'cache_kernel_version':plan['cache']['kernel_version'],'config':plan['config'],'config_sha256':plan['config_sha256'],'inputs_sha256':plan['cache']['inputs_sha256'],'cache_parent_config_sha256':plan['cache']['parent_config_sha256'],'experiment_source_sha256':plan['experiment_source_sha256'],'backbone':run['backbone'],'schedule':run['schedule'],'run_id':run['id'],'dataset_sources':[],'execution_approved':True,'expected_dependencies':plan['expected_dependencies']}
    sources=[plan['cache']['kernel_version'].rsplit('/',1)[0]]
    if run['status']=='paused':
        previous=ROOT/run['attempts'][-1]['bundle'];report,folder,_=read_stage(previous)
        if report['status']!='paused':raise ValueError('Only verified paused epoch-boundary checkpoints may resume')
        spec['resume']={'kernel_version':run['attempts'][-1]['kernel_version'],'progress_sha256':file_hash(folder/'progress.json')};sources.append(spec['resume']['kernel_version'].rsplit('/',1)[0])
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT/'src').rglob('*.py')):z.writestr(str(p.relative_to(ROOT)),p.read_bytes())
        z.writestr('configs/phase1.toml',(ROOT/'configs/phase1.toml').read_bytes())
    code="import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\n"+f"import base64,zipfile,io,sys,json\nfrom pathlib import Path\nz=Path('controlled-code');z.mkdir()\nzipfile.ZipFile(io.BytesIO(base64.b64decode({base64.b64encode(archive.getvalue()).decode()!r}))).extractall(z)\nsys.path.insert(0,str(z/'src'))\nfrom trustretina.runner.controlled_runtime import run\nrun(json.loads({json.dumps(spec)!r}),input_root='/kaggle/input',output='full',nonce={nonce!r},source_commit={plan['source_commit']!r})\n"
    (bundle/'execute.py').write_text(code);kernel='saisathwikregula/trustretina-round2-v9-'+run['id'].replace('_','-')+'-'+nonce[:8]
    write(bundle/'kernel-metadata.json',{'id':kernel,'title':kernel.split('/')[1],'code_file':'execute.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'dataset_sources':[],'competition_sources':[],'kernel_sources':sources,'docker_image':plan['docker_image'],'docker_image_pinning_type':'original'})
    receipt={'kernel_id':kernel,'nonce':nonce,'source_commit':plan['source_commit'],'source_sha256':file_hash(bundle/'execute.py'),'training':True,'gpu_requested':True,'submission_attempted':False,'execution_timeout_seconds':1800,'budget_reservation_seconds':reserved,'plan_sha256':plan['plan_sha256'],'approval_record_sha256':canonical_hash(approval)}
    write(bundle/'submission.json',receipt);run['attempts'].append({'bundle':str(bundle.relative_to(ROOT)),'kernel_id':kernel,'status':'prepared','reserved_seconds':reserved});run['status']='prepared';write(BASE/'registry.json',plan);return bundle,receipt


def advance(plan,api,*,submit=False,approval=None):
    if submit:require_approval(plan,approval)
    verify_plan(plan)
    plan['queue_blocker']=None if approval else 'Awaiting explicit user approval'
    for run in plan['runs']:
        if run['status'] not in ('running','submitting','ambiguous'):continue
        attempt=run['attempts'][-1];bundle=ROOT/attempt['bundle'];receipt=json.loads((bundle/'submission.json').read_text())
        if not receipt.get('kernel_version'):
            plan['queue_blocker']='Ambiguous attempted submission: recover exact remote version/source/nonce; never resubmit';return
        attempt['remote_status']=str(api.kernels_status(receipt['kernel_version']).status).split('.')[-1]
        if attempt['remote_status'] in ('QUEUED','RUNNING'):return
        report,folder,v=transport.retrieve(api,bundle)
        if report['status']=='complete' and (not report.get('strict_best_reload') or not {'best.pt','last.pt','validation-patients.npz','validation-images.npz','history.json'}<=set(report['output_hashes'])):raise ValueError('Incomplete run finalization')
        run.update(status=report['status'],completed_epochs=report.get('completed_epochs'),metrics=report.get('metrics'),report_sha256=v['artifact_sha256'],elapsed_seconds=report['elapsed_seconds']);attempt.update(status=report['status'],kernel_version=receipt['kernel_version'],elapsed_seconds=report['elapsed_seconds'])
        run['total_worker_seconds']=sum(a.get('elapsed_seconds',0) for a in run['attempts'])
        if run['status'] not in ('complete','paused'):plan['queue_blocker']='Failed run preserved; explicit evidence review required before any continuation';return
    if not submit:return
    run=next((r for r in plan['runs'] if r['status']!='complete'),None)
    if run is None:return
    if run['status'] not in ('queued','paused','prepared'):raise ValueError('Unresolved failed/ambiguous state blocks queue')
    known={a['kernel_id'] for r in plan['runs'] for a in r['attempts']}
    unknown={ref for ref in mine(api) if 'trustretina-round2-v9-' in ref}-known
    if unknown:raise ValueError('Unregistered v9 remote jobs found; recover instead of submitting')
    q=quota(api)
    if 'quota_baseline' not in plan:plan['quota_baseline']={'used_seconds':q['used_seconds'],'refresh_time':q['refresh_time']}
    observed=q['used_seconds']-plan['quota_baseline']['used_seconds'] if q['refresh_time']==plan['quota_baseline']['refresh_time'] else 0
    spent=max(sum(a['reserved_seconds'] for r in plan['runs'] for a in r['attempts']),observed)
    next_budget=max(spent,observed+2400) if run['status']=='prepared' else spent+2400
    if next_budget>14400:
        plan['queue_blocker']='Four-hour GPU quota/reservation budget reached; preserve queue';return
    if not q['paid_scaling_disabled'] or q['free_remaining_seconds']<2400:
        plan['queue_blocker']='Free quota insufficient for next half-hour reservation or paid scaling enabled; no fallback';return
    if run['status']=='prepared':
        bundle=ROOT/run['attempts'][-1]['bundle'];receipt=json.loads((bundle/'submission.json').read_text())
    else:bundle,receipt=build(plan,run,approval)
    if receipt['submission_attempted']:raise ValueError('Attempted submission cannot be retried blindly')
    receipt['submission_attempted']=True;write(bundle/'submission.json',receipt);run['status']='submitting';write(BASE/'registry.json',plan)
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):result=api.kernels_push(str(bundle),timeout='1800',acc='NvidiaTeslaT4')
        if not result.version_number:raise ValueError('No version returned; reconcile without another submission')
        receipt.update(kernel_version=receipt['kernel_id']+'/'+str(result.version_number),kernel_version_number=result.version_number,submission_verified=True);write(bundle/'submission.json',receipt)
        run['status']='running';run['attempts'][-1].update(status='running',kernel_version=receipt['kernel_version']);plan['execution_authorized']=True
    except Exception as error:
        run['status']='ambiguous';receipt['submission_error_type']=type(error).__name__;write(bundle/'submission.json',receipt);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','restore','status','check-quota','advance','compare']);parser.add_argument('--submit',action='store_true');parser.add_argument('--approval-file',type=Path);args=parser.parse_args()
    approval=json.loads(args.approval_file.read_text()) if args.approval_file else None
    if args.action=='prepare':plan=prepare()
    elif args.action=='restore':plan=restore()
    else:plan=json.loads((BASE/'registry.json').read_text())
    if args.submit:require_approval(plan,approval)  # before API client or any mutation
    try:
        if args.action=='advance':advance(plan,transport.api_client(),submit=args.submit,approval=approval)
        if args.action=='check-quota':print(json.dumps(quota(transport.api_client()),indent=2))
        if args.action=='compare':
            if any(r['status']!='complete' for r in plan['runs']):raise ValueError('All four completed runs required; no partial winner')
            reports={}
            for r in plan['runs']:
                report,_,_=read_stage(ROOT/r['attempts'][-1]['bundle']);reports[r['id']]=report['metrics']
            from trustretina.training.comparison import select_backbone
            order=[r['id'] for r in plan['runs']];winner=select_backbone(order,reports)
            if winner is None:raise ValueError('All-eight AUROCs required')
            write(ROOT/'results/round2-v9/comparison.json',{'reports':reports,'selected_run':winner,'plan_sha256':plan['plan_sha256'],'development_only':True,'independent_evaluation_required':True})
    finally:write(BASE/'registry.json',plan);export(plan)
    print(json.dumps({'plan_sha256':plan['plan_sha256'],'runs':[{k:r[k] for k in ('id','backbone','schedule','status')} for r in plan['runs']],'reserved_seconds':sum(a['reserved_seconds'] for r in plan['runs'] for a in r['attempts']),'blocker':plan['queue_blocker']},indent=2))


if __name__=='__main__':
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
