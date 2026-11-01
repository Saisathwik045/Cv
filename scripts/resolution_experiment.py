#!/usr/bin/env python3
"""Frozen v13 queue. CPU prepare/read-only status; training needs new bound approval."""
import argparse,base64,contextlib,fcntl,io,json,re,shutil,subprocess,sys,time,tomllib,uuid,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trustretina.data.validation import file_hash,canonical_hash
from trustretina.training.nested import audit_manifest
from resolution_study import MATRIX,validate_config,require_approval
import full_experiment as transport
BASE=ROOT/'artifacts/resolution-v13/experiment';CONFIG=ROOT/'configs/resolution-v13.toml';DOC=ROOT/'docs/RESOLUTION_V13.md'
WORKER_FILES=sorted([str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py')]+['scripts/resolution_runtime.py','scripts/resolution_study.py'])
IDENTITY=('config','config_sha256','protocol_sha256','experiment_source_sha256','controller_sha256','reporter_sha256','source_commit','cache','fold_manifest_sha256','expected_dependencies','docker_image')
write=transport.write


def source_hash():
    import hashlib
    return hashlib.sha256(b''.join(n.encode()+(ROOT/n).read_bytes() for n in WORKER_FILES)).hexdigest()


def seconds(v):
    m=re.fullmatch(r'(\d+(?:\.\d+)?)(?:\.0)?s',str(v))
    if not m:raise ValueError('Unknown quota duration')
    return float(m[1])


def quota(api):
    raw=json.loads(str(api.quota_view()));q=raw['gpuQuota']
    return {'observed_unix':time.time(),'raw':raw,'used_seconds':seconds(q['timeUsed']),
            'reserved_seconds':seconds(q['timeReserved']), 'refresh_time':raw.get('quotaRefreshTime'),
            'free_seconds':seconds(q['totalTimeAllowed'])-seconds(q['timeUsed'])-seconds(q['timeReserved']),
            'paid_disabled':q.get('isPayToScaleEnabled') is False}


def export(p):
    receipts={a['bundle']:json.loads((ROOT/a['bundle']/'submission.json').read_text()) for r in p['runs'] for a in r['attempts']}
    write(ROOT/'results/resolution-v13/queue.json',{'registry':p,'receipts':receipts})


def verify_plan(p):
    validate_config(p['config'])
    if canonical_hash({k:p[k] for k in IDENTITY})!=p['plan_sha256']:raise ValueError('Frozen plan identity changed')
    actual=(file_hash(CONFIG),file_hash(DOC),source_hash(),file_hash(Path(__file__)),file_hash(ROOT/'scripts/report_resolution_results.py'))
    expected=tuple(p[k] for k in ('config_sha256','protocol_sha256','experiment_source_sha256','controller_sha256','reporter_sha256'))
    if actual!=expected:raise ValueError('Frozen source/config/protocol changed; new review/approval required')
    if [(r['id'],r['outer_fold'],r['image_size']) for r in p['runs']]!=list(MATRIX):raise ValueError('Matrix changed')
    m=json.loads((BASE/'fold-manifest.json').read_text())
    rows_path=ROOT/p['cache']['rows_path']
    if file_hash(rows_path)!=p['cache']['rows_sha256'] or m['sha256']!=p['fold_manifest_sha256']:raise ValueError('Train/fold identity changed')
    audit_manifest(json.loads(rows_path.read_text()),m,p['config']);return m


def prepare():
    if (BASE/'registry.json').exists():raise ValueError('Existing plan: reuse/resume, never recreate')
    cfg=validate_config(tomllib.loads(CONFIG.read_text()))
    from trustretina.runner.verified_artifacts import read_stage
    bundle=ROOT/'artifacts/resolution-v13/preparation';report,folder,v=read_stage(bundle)
    if report['status']!='complete' or report['gpu_requested'] or report['training'] or report['parity_mismatched_images'] or not report['new_320_from_native'] or report['native_images_verified']!=4550:
        raise ValueError('Native provenance and exact 224 parity prerequisites incomplete')
    inputs=json.loads((folder/'inputs.json').read_text())
    old=json.loads((ROOT/'artifacts/runner/nested-v11/registry.json').read_text());manifest=json.loads((ROOT/'artifacts/runner/nested-v11/fold-manifest.json').read_text())
    if inputs['fold_manifest_sha256']!=old['fold_manifest_sha256'] or inputs['rows_sha256']!=old['cache']['train_rows_sha256'] or not inputs['parity_exact']:
        raise ValueError('Input/fold assignment identity changed')
    rows=json.loads((folder/inputs['rows_file']).read_text());audit_manifest(rows,manifest,cfg)
    BASE.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'artifacts/runner/nested-v11/fold-manifest.json',BASE/'fold-manifest.json')
    p={'schema_version':13,'config':cfg,'config_sha256':file_hash(CONFIG),'protocol_sha256':file_hash(DOC),
       'experiment_source_sha256':source_hash(),'controller_sha256':file_hash(Path(__file__)),'reporter_sha256':file_hash(ROOT/'scripts/report_resolution_results.py'),
       'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'cache':{'bundle':str(bundle.relative_to(ROOT)),'kernel_version':v['kernel_version'],'inputs_sha256':file_hash(folder/'inputs.json'),
                'rows_path':str((folder/inputs['rows_file']).relative_to(ROOT)),'rows_sha256':inputs['rows_sha256'],'release_sha256':inputs['release_sha256'],
                'report_sha256':v['artifact_sha256'],'resolutions':inputs['resolutions'],'initialization':inputs['weights']['convnext_tiny']},
       'fold_manifest_sha256':manifest['sha256'],'expected_dependencies':old['expected_dependencies'],'docker_image':old['docker_image']}
    p['plan_sha256']=canonical_hash({k:p[k] for k in IDENTITY})
    p.update(runs=[{'id':n,'outer_fold':f,'image_size':s,'status':'queued','attempts':[]} for n,f,s in MATRIX],queue_blocker='New exact training approval required',execution_authorized=False)
    write(BASE/'registry.json',p);export(p);return p


def owned_inventory(api):
    refs=[]
    for page in range(1,21):
        jobs=list(api.kernels_list(mine=True,page=page,page_size=100,sort_by='dateRun'));refs.extend(k.ref for k in jobs)
        if len(jobs)<100:return refs
    raise ValueError('Remote inventory bound exceeded')


def next_budget(p,q,r):
    attempts=[a for run in p['runs'] for a in run['attempts']]
    if len(attempts)>=6:raise ValueError('Six session cap reached; no seventh session')
    if not q['paid_disabled'] or q['reserved_seconds']>0:raise ValueError('Paid/reserved accelerator quota blocks submission')
    reserve=p['config'][f"timeout_{r['image_size']}_seconds"]+300
    if not attempts and q['free_seconds']<21600:raise ValueError('Full six-hour free reservation unavailable for first session; wait/recheck, no partial start')
    charged=q['used_seconds']-p.get('quota_baseline',{}).get('used_seconds',q['used_seconds'])
    if charged<0:raise ValueError('Quota counter moved backwards; reconcile before submission')
    if 'quota_baseline' in p and q['refresh_time']!=p['quota_baseline']['refresh_time']:raise ValueError('Quota window changed; reconcile')
    # Stop rather than assume the requested accelerator has the old charging rate.
    rate=1
    if attempts:
        latest=attempts[-1]
        import math
        rate=max(1,math.ceil(max(0,q['used_seconds']-latest['quota_used_before']-300)/max(1,latest.get('elapsed_seconds',1))))
    reserve=max(reserve,p['config'][f"timeout_{r['image_size']}_seconds"]*rate+300)
    spent=sum(a['reserved_seconds'] for a in attempts)
    if max(spent,charged)+reserve>21600 or q['free_seconds']<reserve:raise ValueError('Actual quota/rate cannot cover next reservation within cap')
    return reserve


def build(p,r,approval,reserve):
    require_approval(p,approval);manifest=verify_plan(p);nonce=uuid.uuid4().hex
    bundle=BASE/r['id']/f"attempt-{len(r['attempts'])+1}";bundle.mkdir(parents=True,exist_ok=False)
    spec={'config':p['config'],'config_sha256':p['config_sha256'],'release_sha256':p['cache']['release_sha256'],
          'plan_sha256':p['plan_sha256'],'protocol_sha256':p['protocol_sha256'],'experiment_source_sha256':p['experiment_source_sha256'],
          'inputs_sha256':p['cache']['inputs_sha256'],'run_id':r['id'],'image_size':r['image_size'],'outer_fold':r['outer_fold'],
          'fold_manifest':manifest,'fold_manifest_sha256':p['fold_manifest_sha256'],'dataset_sources':[],'execution_approved':True,'expected_dependencies':p['expected_dependencies']}
    sources=[p['cache']['kernel_version']]
    if r['status']=='paused':
        previous=ROOT/r['attempts'][-1]['bundle'];v=json.loads((previous/'verification.json').read_text());report=previous/'download'/v['report_relative_path'];progress=report.parent/'progress.json'
        spec['resume']={'kernel_version':v['kernel_version'],'progress_sha256':file_hash(progress)};sources.append(v['kernel_version'])
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in WORKER_FILES:z.write(ROOT/name,name)
    code="import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\n"+f"import base64,zipfile,io,sys,json\nfrom pathlib import Path\nz=Path('resolution-code');z.mkdir()\nzipfile.ZipFile(io.BytesIO(base64.b64decode({base64.b64encode(archive.getvalue()).decode()!r}))).extractall(z)\nsys.path[:0]=[str(z/'src'),str(z/'scripts')]\nfrom resolution_runtime import run\nrun(json.loads({json.dumps(spec)!r}),input_root='/kaggle/input',output='full',nonce={nonce!r},source_commit={p['source_commit']!r})\n"
    (bundle/'execute.py').write_text(code);kernel=f"saisathwikregula/trustretina-res-v13-f{r['outer_fold']}-{r['image_size']}-{nonce[:8]}"
    # Kernel source versions are checked by exact content hashes even if provider metadata normalizes a ref.
    write(bundle/'kernel-metadata.json',{'id':kernel,'title':kernel.split('/')[1],'code_file':'execute.py','language':'python','kernel_type':'script','is_private':True,
          'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'dataset_sources':[],'competition_sources':[],
          'kernel_sources':[s.rsplit('/',1)[0] for s in sources],'docker_image':p['docker_image'],'docker_image_pinning_type':'original'})
    receipt={'kernel_id':kernel,'nonce':nonce,'source_commit':p['source_commit'],'source_sha256':file_hash(bundle/'execute.py'),'training':True,'gpu_requested':True,
             'submission_attempted':False,'execution_timeout_seconds':p['config'][f"timeout_{r['image_size']}_seconds"],
             'budget_reservation_seconds':reserve,'plan_sha256':p['plan_sha256'],'approval_record_sha256':canonical_hash(approval)}
    write(bundle/'submission.json',receipt);r['attempts'].append({'bundle':str(bundle.relative_to(ROOT)),'kernel_id':kernel,'status':'prepared','reserved_seconds':reserve})
    r['status']='prepared';write(BASE/'registry.json',p);return bundle,receipt


def advance(p,api,*,submit=False,approval=None):
    if submit:require_approval(p,approval)
    verify_plan(p)
    if time.time()-p.get('last_advance_unix',0)<180:raise ValueError('Minimum180-second reconciliation interval')
    p['last_advance_unix']=time.time()
    for r in p['runs']:
        if r['status'] not in ('running','submitting','ambiguous'):continue
        a=r['attempts'][-1];receipt=json.loads((ROOT/a['bundle']/'submission.json').read_text())
        if not receipt.get('kernel_version'):raise ValueError('Ambiguous attempt: recover version, never resubmit')
        a['remote_status']=str(api.kernels_status(receipt['kernel_version']).status).split('.')[-1]
        if a['remote_status'] in ('RUNNING','QUEUED'):return
        report,_,v=transport.retrieve(api,ROOT/a['bundle'])
        if report['plan_sha256']!=p['plan_sha256'] or report['run_id']!=r['id']:raise ValueError('Wrong retrieved report')
        if report['status']=='complete' and (not report.get('strict_best_reload') or not report.get('outer_assessed')):raise ValueError('Incomplete outer finalization: recover, do not retrain')
        if report['status']=='paused' and report.get('outer_assessed'):raise ValueError('Paused fit exposed outer patients')
        a.update(status=report['status'],kernel_version=v['kernel_version'],elapsed_seconds=report['elapsed_seconds']);r['status']=report['status']
        if r['status'] not in ('complete','paused'):raise ValueError('Failed attempt preserved; explicit review required')
    if not submit:return
    r=next((r for r in p['runs'] if r['status']!='complete'),None)
    if r is None:p['queue_blocker']=None;return
    if r['status'] not in ('queued','paused'):raise ValueError('Prepared/ambiguous state requires receipt recovery, never blindly push')
    known={a['kernel_id'] for run in p['runs'] for a in run['attempts']}
    if {ref for ref in owned_inventory(api) if 'trustretina-res-v13-f' in ref}-known:raise ValueError('Unregistered remote training job; recover')
    q=quota(api);write(BASE/'quota-latest.json',q);reserve=next_budget(p,q,r)
    if 'quota_baseline' not in p:p['quota_baseline']={'used_seconds':q['used_seconds'],'refresh_time':q['refresh_time']}
    bundle,receipt=build(p,r,approval,reserve)
    write(bundle/'quota-before.json',q);r['attempts'][-1]['quota_used_before']=q['used_seconds']
    receipt['submission_attempted']=True;write(bundle/'submission.json',receipt);r['status']='submitting';write(BASE/'registry.json',p)
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            result=api.kernels_push(str(bundle),timeout=str(receipt['execution_timeout_seconds']),acc='NvidiaTeslaT4')
        if not result.version_number:raise ValueError('Missing provider version; recover')
        receipt.update(kernel_version=receipt['kernel_id']+'/'+str(result.version_number),submission_verified=True);write(bundle/'submission.json',receipt)
        r['status']='running';r['attempts'][-1].update(status='running',kernel_version=receipt['kernel_version']);p['execution_authorized']=True;p['queue_blocker']=None
    except Exception as e:
        r['status']='ambiguous';receipt['submission_error_type']=type(e).__name__;write(bundle/'submission.json',receipt);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','status','check-quota','advance','retrieve']);parser.add_argument('--submit',action='store_true');parser.add_argument('--approval-file',type=Path);a=parser.parse_args()
    approval=json.loads(a.approval_file.read_text()) if a.approval_file else None
    if a.submit and not approval:raise PermissionError('New training approval required before API access')
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=prepare() if a.action=='prepare' else json.loads((BASE/'registry.json').read_text())
        try:
            if a.action=='check-quota':print(json.dumps(quota(transport.api_client()),indent=2))
            if a.action=='advance':
                if a.submit:require_approval(p,approval)
                advance(p,transport.api_client(),submit=a.submit,approval=approval)
            if a.action=='retrieve':
                api=transport.api_client()
                for r in p['runs']:
                    for attempt in r['attempts']:
                        if attempt.get('status') in ('complete','paused','failed'):transport.retrieve(api,ROOT/attempt['bundle'])
        finally:write(BASE/'registry.json',p);export(p)
        print(json.dumps({'plan_sha256':p['plan_sha256'],'runs':p['runs'],'blocker':p['queue_blocker']},indent=2))


if __name__=='__main__':main()
