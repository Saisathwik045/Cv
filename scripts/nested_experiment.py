#!/usr/bin/env python3
"""V11 bounded queue. prepare is CPU-only; submission requires new bound approval."""
import argparse,base64,contextlib,datetime,fcntl,io,json,subprocess,sys,time,tomllib,uuid,zipfile,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import full_experiment as transport
from trustretina.training.nested import validate_config,build_manifest,audit_manifest,require_approval,reserve_attempt,MATRIX
from trustretina.data.validation import file_hash,canonical_hash
from trustretina.runner.verified_artifacts import read_stage
BASE=ROOT/'artifacts/runner/nested-v11';CONFIG=ROOT/'configs/nested-loss-v11.toml';DOC=ROOT/'docs/NESTED_LOSS_V11.md'
write=transport.write
IDENTITY=('config','config_sha256','protocol_sha256','experiment_source_sha256','controller_sha256','reporter_sha256','source_commit','cache','fold_manifest_sha256','expected_dependencies','docker_image')


def quota(api):
    raw=json.loads(str(api.quota_view()));q=raw['gpuQuota']
    def seconds(v):
        m=re.fullmatch(r'(\d+(?:\.\d+)?)(?:\.0)?s',str(v))
        if not m:raise ValueError('Unknown quota duration')
        return float(m.group(1))
    r={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'raw':raw,
       'used_seconds':seconds(q['timeUsed']),'refresh_time':raw.get('quotaRefreshTime'),
       'free_remaining_seconds':seconds(q['totalTimeAllowed'])-seconds(q['timeUsed'])-seconds(q['timeReserved']),
       'paid_scaling_disabled':q.get('isPayToScaleEnabled') is False}
    write(BASE/'latest-quota.json',r);return r


def export(p):
    receipts={a['bundle']:json.loads((ROOT/a['bundle']/'submission.json').read_text()) for r in p['runs'] for a in r['attempts']}
    write(ROOT/'results/nested-v11/queue.json',{'registry':p,'receipts':receipts})


def prepare():
    if (BASE/'registry.json').exists():raise ValueError('Existing plan: resume/reuse, never recreate')
    cfg=validate_config(tomllib.loads(CONFIG.read_text()))
    subprocess.run(['git','diff','--exit-code','HEAD','--','src','scripts/nested_experiment.py','configs/nested-loss-v11.toml','docs/NESTED_LOSS_V11.md'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    tracked=['scripts/nested_experiment.py','configs/nested-loss-v11.toml','docs/NESTED_LOSS_V11.md']+[str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py')]
    subprocess.run(['git','ls-files','--error-unmatch',*tracked],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    old=json.loads((ROOT/'results/round2-v9/queue.json').read_text())['registry'];cache=old['cache']
    folder=ROOT/cache['bundle']/'download/prepared';index=folder/'inputs.json'
    # Receipt identities and index + TRAIN row digest only. Do not read old validation payloads.
    verification=json.loads((ROOT/cache['bundle']/'verification.json').read_text())
    submission=json.loads((ROOT/cache['bundle']/'submission.json').read_text())
    if not verification.get('integrity_verified') or verification['kernel_version']!=cache['kernel_version'] or submission['nonce']!=verification['nonce'] or file_hash(index)!=cache['inputs_sha256']:
        raise ValueError('Pinned cache provenance/index changed')
    inputs=json.loads(index.read_text());item=inputs['roles']['train'];rows_path=folder/item['rows_file']
    if not rows_path.resolve().is_relative_to(folder.resolve()) or file_hash(rows_path)!=item['rows_sha256']:raise ValueError('Train row checksum mismatch')
    rows=json.loads(rows_path.read_text());manifest=build_manifest(rows,cfg,item['rows_sha256'])
    if manifest['training']['patients']!=2275 or manifest['training']['components']!=2240:raise ValueError('Wrong source cohort')
    write(BASE/'fold-manifest.json',manifest)
    plan={'config':cfg,'config_sha256':file_hash(CONFIG),'protocol_sha256':file_hash(DOC),
          'experiment_source_sha256':transport.source_hash(),'controller_sha256':file_hash(Path(__file__)),'reporter_sha256':file_hash(ROOT/'scripts/report_nested_results.py'),
          'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'cache':{**cache,'parent_config_sha256':inputs['config_sha256'],'release_sha256':inputs['release_sha256'],
                   'train_rows_sha256':item['rows_sha256'],'training_metadata_file':str(rows_path.relative_to(ROOT))},
          'fold_manifest_sha256':manifest['sha256'],'expected_dependencies':old['expected_dependencies'],'docker_image':old['docker_image']}
    plan.update(plan_sha256=canonical_hash({k:plan[k] for k in IDENTITY}),runs=[{'id':i,'outer_fold':k,'arm':a,'status':'queued','attempts':[]} for i,k,a in MATRIX],queue_blocker='Awaiting explicit six-fit / four-hour approval',execution_authorized=False)
    write(BASE/'registry.json',plan);export(plan)
    histories=json.loads((ROOT/'results/round2-v9/verified-comparison.json').read_text())['comparison']
    ref=next(r for r in histories if r['run']=='convnext_cosine');epoch_seconds=ref['worker_seconds']/ref['epochs_completed']
    comparison=json.loads((ROOT/'results/round2-v9/verified-comparison.json').read_text())
    ref_run=next(r for r in old['runs'] if r['id']=='convnext_cosine')
    history_path=ROOT/ref_run['attempts'][-1]['bundle']/'download/full/history.json'
    if file_hash(history_path)!=comparison['sources']['convnext_cosine']['history_sha256']:raise ValueError('Reference timing history changed')
    history=json.loads(history_path.read_text())
    compute=sum(r['epoch_seconds'] for r in history)/len(history)
    validation=sum(r['validation_seconds'] for r in history)/len(history)
    fixed=max(0,epoch_seconds-compute)
    estimates=[]
    for f in manifest['folds']:
        s=f['summary'];scaled=(compute-validation)*s['fit']['patients']/2275+validation*s['inner']['patients']/525+fixed
        outer_seconds=2*s['outer']['patients']*.004763
        estimates.append({'outer_fold':f['outer_fold'],'seconds_per_epoch_estimate':scaled,'max_epoch_worker_seconds_per_fit_estimate':25*scaled+outer_seconds+180})
    total=2*sum(e['max_epoch_worker_seconds_per_fit_estimate'] for e in estimates)
    write(ROOT/'results/nested-v11/preparation.json',{'status':'CPU preparation complete; no training submitted','plan_sha256':plan['plan_sha256'],
        'fold_manifest_sha256':manifest['sha256'],'training':manifest['training'],'folds':[{'outer_fold':f['outer_fold'],'roles':f['summary']} for f in manifest['folds']],
        'measured_reference_worker_seconds':ref['worker_seconds'],'measured_reference_epochs':ref['epochs_completed'],'measured_compute_epoch_seconds':compute,'measured_selection_epoch_seconds':validation,'retained_fixed_overhead_seconds_per_epoch':fixed,
        'estimates':estimates,'estimated_six_fit_worker_hours_at_25_epochs':total/3600,'estimate_includes_per_fit_cold_start_io_seconds':180,
        'hard_total_quota_seconds_including_retries':14400,'maximum_submitted_sessions_including_resumes':6,'maximum_session_seconds':2100,'overhead_guard_per_session':300,
        'runs_submitted':0,'historical_final_opened':False,'old_validation_payload_opened':False,'calibration_accessed':False})
    return plan


def restore():
    if (BASE/'registry.json').exists():raise ValueError('Existing registry: refusing overwrite')
    saved=json.loads((ROOT/'results/nested-v11/queue.json').read_text());p=saved['registry']
    rows=json.loads((ROOT/p['cache']['training_metadata_file']).read_text())
    m=build_manifest(rows,p['config'],p['cache']['train_rows_sha256'])
    if m['sha256']!=p['fold_manifest_sha256']:raise ValueError('Restored assignments differ; no alternate split permitted')
    write(BASE/'fold-manifest.json',m)
    for relative,receipt in saved['receipts'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(BASE.resolve()):raise ValueError('Unsafe receipt path')
        if (path/'submission.json').exists() and json.loads((path/'submission.json').read_text())!=receipt:raise ValueError('Conflicting saved receipt')
        write(path/'submission.json',receipt)
    verify_plan(p);write(BASE/'registry.json',p);return p


def verify_plan(p):
    validate_config(p['config'])
    if canonical_hash({k:p[k] for k in IDENTITY})!=p['plan_sha256']:raise ValueError('Plan identity changed')
    if file_hash(CONFIG)!=p['config_sha256'] or file_hash(DOC)!=p['protocol_sha256'] or file_hash(Path(__file__))!=p['controller_sha256'] or file_hash(ROOT/'scripts/report_nested_results.py')!=p['reporter_sha256'] or transport.source_hash()!=p['experiment_source_sha256']:raise ValueError('Source/config/protocol changed; new review/approval required')
    if [(r['id'],r['outer_fold'],r['arm']) for r in p['runs']]!=list(MATRIX):raise ValueError('Run matrix changed')
    m=json.loads((BASE/'fold-manifest.json').read_text())
    rows_path=ROOT/p['cache']['training_metadata_file']
    if file_hash(rows_path)!=p['cache']['train_rows_sha256'] or m['sha256']!=p['fold_manifest_sha256']:raise ValueError('Train/fold metadata changed')
    audit_manifest(json.loads(rows_path.read_text()),m,p['config'])
    return m


def build(p,r,approval):
    require_approval(p,approval);manifest=verify_plan(p);reserve=reserve_attempt(p,r)
    nonce=uuid.uuid4().hex;bundle=BASE/r['id']/f'attempt-{len(r["attempts"])+1}';bundle.mkdir(parents=True,exist_ok=False)
    spec={'config':p['config'],'config_sha256':p['config_sha256'],'release_sha256':p['cache']['release_sha256'],
        'plan_sha256':p['plan_sha256'],'protocol_sha256':p['protocol_sha256'],'experiment_source_sha256':p['experiment_source_sha256'],
        'inputs_sha256':p['cache']['inputs_sha256'],'cache_parent_config_sha256':p['cache']['parent_config_sha256'],
        'run_id':r['id'],'arm':r['arm'],'outer_fold':r['outer_fold'],'fold_manifest':manifest,'fold_manifest_sha256':p['fold_manifest_sha256'],
        'dataset_sources':[],'execution_approved':True,'expected_dependencies':p['expected_dependencies']}
    sources=[p['cache']['kernel_version'].rsplit('/',1)[0]]
    if r['status']=='paused':
        previous=ROOT/r['attempts'][-1]['bundle'];report,folder,_=read_stage(previous)
        if report['status']!='paused' or report.get('outer_assessed'):raise ValueError('Only unassessed epoch-boundary pause may resume')
        spec['resume']={'kernel_version':r['attempts'][-1]['kernel_version'],'progress_sha256':file_hash(folder/'progress.json')};sources.append(spec['resume']['kernel_version'].rsplit('/',1)[0])
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted((ROOT/'src').rglob('*.py')):z.writestr(str(path.relative_to(ROOT)),path.read_bytes())
        z.writestr('configs/phase1.toml',(ROOT/'configs/phase1.toml').read_bytes())
    code="import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\n"+f"import base64,zipfile,io,sys,json\nfrom pathlib import Path\nz=Path('nested-code');z.mkdir()\nzipfile.ZipFile(io.BytesIO(base64.b64decode({base64.b64encode(archive.getvalue()).decode()!r}))).extractall(z)\nsys.path.insert(0,str(z/'src'))\nfrom trustretina.runner.nested_runtime import run\nrun(json.loads({json.dumps(spec)!r}),input_root='/kaggle/input',output='full',nonce={nonce!r},source_commit={p['source_commit']!r})\n"
    (bundle/'execute.py').write_text(code);kernel='saisathwikregula/trustretina-nested-v11-'+r['id'].replace('_','-')+'-'+nonce[:8]
    write(bundle/'kernel-metadata.json',{'id':kernel,'title':kernel.split('/')[1],'code_file':'execute.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'dataset_sources':[],'competition_sources':[],'kernel_sources':sources,'docker_image':p['docker_image'],'docker_image_pinning_type':'original'})
    receipt={'kernel_id':kernel,'nonce':nonce,'source_commit':p['source_commit'],'source_sha256':file_hash(bundle/'execute.py'),'training':True,'gpu_requested':True,'submission_attempted':False,'execution_timeout_seconds':2100,'budget_reservation_seconds':reserve,'plan_sha256':p['plan_sha256'],'approval_record_sha256':canonical_hash(approval)}
    write(bundle/'submission.json',receipt);r['attempts'].append({'bundle':str(bundle.relative_to(ROOT)),'kernel_id':kernel,'status':'prepared','reserved_seconds':reserve});r['status']='prepared';write(BASE/'registry.json',p);return bundle,receipt


def inventory(api):
    found=[]
    for page in range(1,21):
        batch=list(api.kernels_list(mine=True,page=page,page_size=100,sort_by='dateRun'));found.extend(k.ref for k in batch)
        if len(batch)<100:return found
    raise ValueError('Inventory bound exceeded; reconcile before submitting')


def advance(p,api,*,submit=False,approval=None):
    if submit:require_approval(p,approval)
    verify_plan(p)
    now=time.time()
    if now-p.get('last_advance_unix',0)<180:raise ValueError('Advance interval is at least 180 seconds')
    p['last_advance_unix']=now
    for r in p['runs']:
        if r['status'] not in ('running','submitting','ambiguous'):continue
        a=r['attempts'][-1];b=ROOT/a['bundle'];receipt=json.loads((b/'submission.json').read_text())
        if not receipt.get('kernel_version'):p['queue_blocker']='Ambiguous submission: recover exact version; never blindly retry';return
        a['remote_status']=str(api.kernels_status(receipt['kernel_version']).status).split('.')[-1]
        if a['remote_status'] in ('RUNNING','QUEUED'):return
        report,_,v=transport.retrieve(api,b)
        if report['run_id']!=r['id'] or report['plan_sha256']!=p['plan_sha256']:raise ValueError('Wrong retrieved run/plan')
        if report['status']=='complete' and (not report.get('strict_best_reload') or not report.get('outer_assessed') or not {'best.pt','last.pt','outer-patients.npz','outer-images.npz','history.json'}<=set(report['output_hashes'])):raise ValueError('Incomplete outer finalization; preserve checkpoint, do not retrain')
        if report['status']=='paused' and report.get('outer_assessed'):raise ValueError('Paused run exposed outer data')
        a.update(status=report['status'],kernel_version=receipt['kernel_version'],elapsed_seconds=report['elapsed_seconds'])
        r.update(status=report['status'],completed_epochs=report.get('completed_epochs'),report_sha256=v['artifact_sha256'])
        if r['status'] not in ('complete','paused'):p['queue_blocker']='Failed run preserved: review exact evidence before continuation';return
    if not submit:return
    r=next((r for r in p['runs'] if r['status']!='complete'),None)
    if r is None:p['queue_blocker']=None;return
    if r['status'] not in ('queued','paused','prepared'):raise ValueError('Unresolved state blocks queue')
    known={a['kernel_id'] for r in p['runs'] for a in r['attempts']}
    existing={ref for ref in inventory(api) if 'trustretina-nested-v11-' in ref}
    if existing-known:raise ValueError('Unregistered matching job: recover before submit')
    if r['status']=='prepared' and r['attempts'][-1]['kernel_id'] in existing:raise ValueError('Prepared ID already remote; recover version, never push again')
    q=quota(api)
    if 'quota_baseline' not in p:p['quota_baseline']={k:q[k] for k in ('used_seconds','refresh_time')}
    if q['refresh_time']!=p['quota_baseline']['refresh_time']:raise ValueError('Quota window changed; review accounting before continuing')
    observed=q['used_seconds']-p['quota_baseline']['used_seconds']
    if observed<0:raise ValueError('Quota counter decreased; review accounting')
    spent=sum(a['reserved_seconds'] for r in p['runs'] for a in r['attempts'])
    need=max(spent,observed+2400) if r['status']=='prepared' else max(spent,observed)+2400
    if need>14400 or not q['paid_scaling_disabled'] or q['free_remaining_seconds']<2400:
        p['queue_blocker']='Four-hour/free-quota guard blocks next session; no paid fallback';return
    if r['status']=='prepared':b=ROOT/r['attempts'][-1]['bundle'];receipt=json.loads((b/'submission.json').read_text())
    else:
        if len(r['attempts'])>=2:
            p['queue_blocker']='Two-attempt fit limit reached; preserve unfinished queue';return
        b,receipt=build(p,r,approval)
    if receipt['submission_attempted']:raise ValueError('Never blindly repeat attempted submission')
    receipt['submission_attempted']=True;write(b/'submission.json',receipt);r['status']='submitting';write(BASE/'registry.json',p)
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):result=api.kernels_push(str(b),timeout='2100',acc='NvidiaTeslaT4')
        if not result.version_number:raise ValueError('Missing remote version; reconcile')
        receipt.update(kernel_version=receipt['kernel_id']+'/'+str(result.version_number),submission_verified=True)
        write(b/'submission.json',receipt);r['status']='running';r['attempts'][-1].update(status='running',kernel_version=receipt['kernel_version']);p['execution_authorized']=True;p['queue_blocker']=None
    except Exception as e:
        r['status']='ambiguous';receipt['submission_error_type']=type(e).__name__;write(b/'submission.json',receipt);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','restore','retrieve','status','check-quota','advance']);parser.add_argument('--submit',action='store_true');parser.add_argument('--approval-file',type=Path);a=parser.parse_args()
    approval=json.loads(a.approval_file.read_text()) if a.approval_file else None
    if a.submit and not approval:raise PermissionError('No new GPU submission without explicit bound approval')
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.action=='prepare':p=prepare()
        elif a.action=='restore':p=restore()
        else:p=json.loads((BASE/'registry.json').read_text())
        try:
            if a.action=='retrieve':
                api=transport.api_client()
                for run in p['runs']:
                    for attempt in run['attempts']:
                        if attempt.get('status') in ('complete','paused','failed'):
                            transport.retrieve(api,ROOT/attempt['bundle'])
            if a.action=='check-quota':print(json.dumps(quota(transport.api_client()),indent=2))
            if a.action=='advance':
                if a.submit:require_approval(p,approval)
                advance(p,transport.api_client(),submit=a.submit,approval=approval)
        finally:write(BASE/'registry.json',p);export(p)
        print(json.dumps({'plan_sha256':p['plan_sha256'],'runs':p['runs'],'blocker':p['queue_blocker']},indent=2))


if __name__=='__main__':main()
