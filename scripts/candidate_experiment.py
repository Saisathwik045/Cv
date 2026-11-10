#!/usr/bin/env python3
"""Frozen v14 queue: CPU freeze/status/recovery; GPU push requires new exact approval."""
import argparse,base64,contextlib,fcntl,io,json,subprocess,sys,time,uuid,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import full_experiment as transport
from trustretina.data.validation import canonical_hash,file_hash
from trustretina.runner.verified_artifacts import read_stage
from candidate_study import MATRIX,validate_config,audit_roles,train_weights,require_approval,next_budget
from resolution_experiment import quota,owned_inventory
BASE=ROOT/'artifacts/candidate-v14/experiment';CONFIG=ROOT/'configs/candidate-v14.json';DOC=ROOT/'docs/CANDIDATE_READY_V14.md'
FILES=sorted([str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py')]+['scripts/resolution_runtime.py','scripts/resolution_study.py','scripts/candidate_runtime.py','scripts/candidate_study.py'])
IDENTITY=('config','config_sha256','worker_source_sha256','controller_sha256','protocol_sha256','source_commit','release_sha256','role_manifest_sha256','inputs_sha256','train_inputs_sha256','sources','expected_dependencies','docker_image','orchestration_dependency_sha256','input_report_sha256')
write=transport.write


def source_hash():
    import hashlib
    return hashlib.sha256(b''.join(n.encode()+(ROOT/n).read_bytes() for n in FILES)).hexdigest()


def verify_plan(p):
    validate_config(p['config'])
    if canonical_hash({k:p[k] for k in IDENTITY})!=p['plan_sha256']:raise ValueError('Frozen v14 plan identity changed')
    if (file_hash(CONFIG),source_hash(),file_hash(Path(__file__)),file_hash(DOC))!=(p['config_sha256'],p['worker_source_sha256'],p['controller_sha256'],p['protocol_sha256']):raise ValueError('Frozen code/config/protocol changed')
    if any(file_hash(ROOT/name)!=digest for name,digest in p['orchestration_dependency_sha256'].items()):raise ValueError('Frozen orchestration/recovery dependency changed')
    if [(r['id'],r['image_size']) for r in p['runs']]!=list(MATRIX):raise ValueError('Two-fit matrix changed')
    return p


def export(p):
    public={k:v for k,v in p.items() if k!='local'}
    receipts={a['bundle']:json.loads((ROOT/a['bundle']/'submission.json').read_text()) for r in p['runs'] for a in r['attempts']}
    write(ROOT/'results/candidate-v14/frozen-plan.json',{'registry':public,'receipts':receipts})


def freeze():
    if (BASE/'registry.json').exists():raise ValueError('Existing frozen registry: use status/recover, do not recreate')
    recovery=json.loads((ROOT/'artifacts/candidate-v14/preparation/recovery.json').read_text());report,folder,v=read_stage(Path(recovery['bundle']))
    if report['status']!='complete' or report['gpu_requested'] or report['training'] or report.get('native_images_verified')!=1750 or report.get('calibration_labels_in_output') is not False:raise ValueError('Actual completed CPU readiness evidence required')
    plan=json.loads((ROOT/'results/candidate-v14/proposed-plan.json').read_text());old=json.loads((ROOT/'results/resolution-v13/queue.json').read_text())['registry'];cfg=validate_config(json.loads(CONFIG.read_text()))
    trainroot=ROOT/old['cache']['bundle']/'download/prepared';trainindex=json.loads((trainroot/'inputs.json').read_text());index=json.loads((folder/'inputs.json').read_text())
    if file_hash(trainroot/'inputs.json')!=old['cache']['inputs_sha256'] or file_hash(folder/'inputs.json')!=report['inputs_sha256']:raise ValueError('Actual index hash changed')
    roles={'train':json.loads((trainroot/trainindex['rows_file']).read_text())}
    if file_hash(trainroot/trainindex['rows_file'])!=trainindex['rows_sha256']:raise ValueError('TRAIN rows changed')
    for size in ('224','320'):
        item=trainindex['resolutions'][size]
        if file_hash(trainroot/item['file'])!=item['sha256']:raise ValueError('Actual TRAIN tensor mismatch')
    for role,item in index['roles'].items():
        if file_hash(folder/item['rows_file'])!=item['rows_sha256'] or item['manifest_sha256']!=plan['partitions'][role]['manifest_sha256']:raise ValueError('Actual row lineage changed')
        roles[role]=json.loads((folder/item['rows_file']).read_text())
        for size in ('224','320'):
            data=item['resolutions'][size];path=folder/data['file']
            if file_hash(path)!=data['sha256']:raise ValueError('Actual role tensor mismatch')
            import numpy as np
            a=np.load(path,mmap_mode='r',allow_pickle=False)
            if a.dtype!=np.uint8 or a.shape!=(len(roles[role]),3,int(size),int(size)):raise ValueError('Actual tensor shape/dtype mismatch')
    counts=audit_roles(roles,{'train':2275,'validation':525,'calibration':350});weights=train_weights(roles['train'])
    import numpy as np
    for role in ('train','validation'):
        unique={r['patient_id']:r['labels'] for r in roles[role]};positive=np.asarray(list(unique.values())).sum(0)
        if positive.tolist()!=[plan['partitions'][role]['positive_patients'][k] for k in 'NDGCAHMO']:raise ValueError('Frozen all-eight support changed')
        counts[role]['positive_patients']=dict(zip('NDGCAHMO',positive.tolist()))
    recovered=json.loads((ROOT/'artifacts/candidate-v14/input-recovery.json').read_text());calroot=Path(recovered['roles']['v6']['folder']);calindex=json.loads((calroot/'inputs.json').read_text())['roles']['calibration'];calfile=calroot/calindex['rows_file']
    if file_hash(calfile)!=calindex['rows_sha256']:raise ValueError('Separate calibration label lineage changed')
    cal=json.loads(calfile.read_text())
    if [{k:v for k,v in r.items() if k!='labels'} for r in cal]!=roles['calibration']:raise ValueError('Unlabeled calibration projection differs')
    positive=np.asarray(list({r['patient_id']:r['labels'] for r in cal}.values())).sum(0)
    if positive.tolist()!=[plan['partitions']['calibration']['positive_patients'][k] for k in 'NDGCAHMO']:raise ValueError('Calibration metadata support differs')
    counts['calibration']['positive_patients']=dict(zip('NDGCAHMO',positive.tolist()))
    p={'schema_version':14,'config':cfg,'config_sha256':file_hash(CONFIG),'worker_source_sha256':source_hash(),'controller_sha256':file_hash(Path(__file__)),'protocol_sha256':file_hash(DOC),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'release_sha256':plan['frozen_v4_release_sha256'],'role_manifest_sha256':{k:x['manifest_sha256'] for k,x in plan['partitions'].items()},'inputs_sha256':file_hash(folder/'inputs.json'),'train_inputs_sha256':old['cache']['inputs_sha256'],'sources':{'train':old['cache']['kernel_version'],'development_roles':v['kernel_version']},'expected_dependencies':old['expected_dependencies'],'docker_image':old['docker_image']}
    p['input_report_sha256']={'train':old['cache']['report_sha256'],'development_roles':v['artifact_sha256']}
    p['orchestration_dependency_sha256']={name:file_hash(ROOT/name) for name in ('scripts/full_experiment.py','scripts/resolution_experiment.py','scripts/candidate_recover.py')}
    p['plan_sha256']=canonical_hash({k:p[k] for k in IDENTITY});p.update(execution_authorized=False,technical_training_inputs_ready=True,external_confirmation_ready=False,maximum_gpu_quota_seconds=14400,maximum_sessions=3,runs=[{'id':name,'image_size':size,'status':'awaiting_approval','attempts':[]} for name,size in MATRIX],local={'train_folder':str(trainroot),'roles_folder':str(folder),'preparation_bundle':recovery['bundle']},role_support=counts,positive_weights=weights.tolist())
    BASE.mkdir(parents=True,exist_ok=True);write(BASE/'registry.json',p);export(p)
    return p


def retrieve_attempt(api,p,run,attempt):
    # Large future outputs use private scratch, never delete the original inputs/checkpoints.
    bundle=ROOT/attempt['bundle'];target=Path('/tmp/trustretina-v14-private/runs')/run['id']/f"attempt-{len(run['attempts'])}";target.mkdir(parents=True,exist_ok=True)
    import shutil
    if not (target/'submission.json').exists():shutil.copyfile(bundle/'submission.json',target/'submission.json')
    report,folder,v=transport.retrieve(api,target)
    if report.get('plan_sha256')!=p['plan_sha256'] or report.get('run_id')!=run['id']:raise ValueError('Retrieved wrong fit/source plan')
    attempt.update(status=report['status'],elapsed_seconds=report['elapsed_seconds'],artifact_bundle=str(target),report_sha256=v['artifact_sha256'])
    if report['status']=='paused':
        state=json.loads((folder/'progress.json').read_text())
        for name,digest in state['hashes'].items():
            if file_hash(folder/name)!=digest:raise ValueError('Paused checkpoint hashes changed')
        attempt.update(resume_verified=True,progress_sha256=file_hash(folder/'progress.json'))
    if report['status']=='complete' and not all(report.get(k) for k in ('strict_best_reload','model_state_invariance','calibration_inference_after_freeze')):
        state=json.loads((folder/'progress.json').read_text())
        for name,digest in state['hashes'].items():
            if file_hash(folder/name)!=digest:raise ValueError('Incomplete post-freeze checkpoint recovery failed')
        attempt.update(status='paused',resume_verified=True,progress_sha256=file_hash(folder/'progress.json'),recovery_reason='Fit ended before complete post-freeze evidence')
        run['status']='paused'
    else:run['status']=report['status']


def build(p,run,approval,timeout,reserve):
    require_approval(p,approval);verify_plan(p);nonce=uuid.uuid4().hex
    bundle=BASE/run['id']/f"attempt-{len(run['attempts'])+1}";bundle.mkdir(parents=True,exist_ok=False)
    spec={k:p[k] for k in ('config','config_sha256','release_sha256','role_manifest_sha256','plan_sha256','worker_source_sha256','inputs_sha256','train_inputs_sha256','expected_dependencies')}
    spec.update(run_id=run['id'],image_size=run['image_size'],session_timeout_seconds=timeout,dataset_sources=[],execution_approved=True)
    sources=list(p['sources'].values())
    if run['attempts']:
        prior=run['attempts'][-1];spec['resume']={'progress_sha256':prior['progress_sha256'],'kernel_version':prior['kernel_version']};sources.append(prior['kernel_version'])
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in FILES:z.write(ROOT/name,name)
    code="import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\n"+f"import base64,zipfile,io,sys,json\nfrom pathlib import Path\np=Path('v14-code');p.mkdir()\nzipfile.ZipFile(io.BytesIO(base64.b64decode({base64.b64encode(archive.getvalue()).decode()!r}))).extractall(p)\nsys.path[:0]=[str(p/'src'),str(p/'scripts')]\nfrom candidate_runtime import run\nrun(json.loads({json.dumps(spec)!r}),input_root='/kaggle/input',output='full',nonce={nonce!r},source_commit={p['source_commit']!r})\n"
    (bundle/'execute.py').write_text(code);kernel=f"saisathwikregula/trustretina-v14-fit-{run['image_size']}-{nonce[:8]}"
    write(bundle/'kernel-metadata.json',{'id':kernel,'title':kernel.split('/')[1],'code_file':'execute.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'dataset_sources':[],'competition_sources':[],'kernel_sources':[s.rsplit('/',1)[0] for s in sources],'docker_image':p['docker_image'],'docker_image_pinning_type':'original'})
    receipt={'kernel_id':kernel,'nonce':nonce,'source_commit':p['source_commit'],'source_sha256':file_hash(bundle/'execute.py'),'training':True,'gpu_requested':True,'plan_sha256':p['plan_sha256'],'submission_attempted':False,'execution_timeout_seconds':timeout,'budget_reservation_seconds':reserve,'approval_record_sha256':canonical_hash(approval)}
    write(bundle/'submission.json',receipt);run['attempts'].append({'bundle':str(bundle.relative_to(ROOT)),'kernel_id':kernel,'status':'prepared','reserved_seconds':reserve});run['status']='prepared'
    return bundle,receipt


def advance(p,api,submit=False,approval=None):
    verify_plan(p)
    if submit:require_approval(p,approval)
    if time.time()-p.get('last_poll_unix',0)<180:raise ValueError('Poll interval180s; inspect saved state')
    p['last_poll_unix']=time.time()
    active=next((r for r in p['runs'] if r['status'] in ('prepared','submitting','running','ambiguous')),None)
    if active:
        a=active['attempts'][-1];receipt=json.loads((ROOT/a['bundle']/'submission.json').read_text())
        if not receipt.get('kernel_version'):raise ValueError('Prepared/ambiguous attempt: reconcile exact provider version; never resubmit')
        status=str(api.kernels_status(receipt['kernel_version']).status).split('.')[-1];a['remote_status']=status
        if status in ('RUNNING','QUEUED'):return
        retrieve_attempt(api,p,active,a);return
    run=next((r for r in p['runs'] if r['status']!='complete'),None)
    if run is None:return
    if run['status']=='failed':raise ValueError('Failed fit preserved; no blind replacement; matrix incomplete')
    if not submit:return
    known={a['kernel_id'] for r in p['runs'] for a in r['attempts']}
    if {r for r in owned_inventory(api) if 'trustretina-v14-fit-' in r}-known:raise ValueError('Unregistered existing v14 job; reconcile first')
    q=quota(api);write(BASE/'quota-latest.json',q);timeout,reserve=next_budget(p,q,run)
    if 'quota_baseline' not in p:p['quota_baseline']={k:q[k] for k in ('used_seconds','refresh_time')}
    bundle,receipt=build(p,run,approval,timeout,reserve);a=run['attempts'][-1];a['quota_used_before']=q['used_seconds'];write(bundle/'quota-before.json',q)
    # Durable reservation/receipt precede API mutation. Any error consumes that slot pending reconciliation.
    receipt['submission_attempted']=True;run['status']='submitting';a['status']='submitting';write(bundle/'submission.json',receipt);write(BASE/'registry.json',p);export(p)
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):result=api.kernels_push(str(bundle),timeout=str(timeout),acc='NvidiaTeslaT4')
        receipt['provider_response_version']=result.version_number;write(bundle/'submission.json',receipt)
        if result.version_number!=1:raise ValueError('Unexpected/ambiguous GPU version; reconcile, never resubmit')
        version=receipt['kernel_id']+'/'+str(result.version_number);receipt.update(kernel_version=version,submission_verified=True);a.update(kernel_version=version,status='running');run['status']='running';p['execution_authorized']=True;write(bundle/'submission.json',receipt)
    except Exception as e:
        run['status']='ambiguous';a['status']='ambiguous';receipt['error_type']=type(e).__name__;write(bundle/'submission.json',receipt);raise



def reconcile(p,api,run_id,version):
    run=next((r for r in p['runs'] if r['id']==run_id),None)
    if not run or not run['attempts']:raise ValueError('Only an already-reserved attempt can be recovered')
    a=run['attempts'][-1];bundle=ROOT/a['bundle'];receipt=json.loads((bundle/'submission.json').read_text())
    if receipt.get('provider_response_version') not in (None,1):raise ValueError('Unexpected multiple provider versions; stop and reconcile accounting/session count')
    if version!=receipt['kernel_id']+'/1':raise ValueError('Only the recorded unique kernel version1 may be reconciled')
    target=bundle/'reconciled-source';target.mkdir(exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):api.kernels_pull(version,path=str(target),metadata=True)
    m=json.loads((target/'kernel-metadata.json').read_text())
    if not m['is_private'] or not m['enable_gpu'] or m['enable_tpu'] or file_hash(target/m['code_file'])!=receipt['source_sha256']:raise ValueError('Existing remote source/resources differ; preserve ambiguity')
    receipt.update(kernel_version=version,submission_verified=True,reconciled_without_submission=True)
    write(bundle/'submission.json',receipt);a.update(kernel_version=version,status='running');run['status']='running'

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','status','check-quota','advance','retrieve','reconcile']);parser.add_argument('--submit',action='store_true');parser.add_argument('--approval-file',type=Path);parser.add_argument('--run-id');parser.add_argument('--kernel-version');args=parser.parse_args()
    if args.submit and not args.approval_file:raise PermissionError('GPU submission needs separate exact approval before API access')
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='freeze':p=freeze()
        else:p=json.loads((BASE/'registry.json').read_text());verify_plan(p)
        approval=json.loads(args.approval_file.read_text()) if args.approval_file else None
        if args.submit:require_approval(p,approval)
        try:
            if args.action=='check-quota':print(json.dumps(quota(transport.api_client()),indent=2))
            if args.action=='reconcile':reconcile(p,transport.api_client(),args.run_id,args.kernel_version)
            if args.action=='advance':advance(p,transport.api_client(),args.submit,approval)
            if args.action=='retrieve':
                api=transport.api_client()
                for r in p['runs']:
                    if r['attempts'] and r['status'] in ('complete','paused','failed'):retrieve_attempt(api,p,r,r['attempts'][-1])
        finally:write(BASE/'registry.json',p);export(p)
        print(json.dumps({'plan_sha256':p['plan_sha256'],'GPU_authorized':p['execution_authorized'],'runs':p['runs']},indent=2))


if __name__=='__main__':main()
