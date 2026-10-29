#!/usr/bin/env python3
"""Sequential private free-Kaggle queue. One status call per advance; no background monitor."""
import argparse, base64, contextlib, fcntl, hashlib, io, json, os, subprocess, sys, time, tomllib, uuid, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trustretina.data.validation import file_hash


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def api_client():
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle import api
    return api


def retrieve(api,bundle):
    from trustretina.runner.verified_artifacts import read_stage
    receipt=json.loads((bundle/'submission.json').read_text())
    # Verified local copies are immutable; do not redownload huge checkpoints every poll.
    if (bundle/'verification.json').exists():return read_stage(bundle)
    token=None
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        while True:
            _,token=api.kernels_output(receipt['kernel_version'],path=str(bundle/'download'),file_pattern=r'^(prepared/.*|full/.*)$',page_token=token,page_size=100)
            if not token:break
        logs=api.kernels_logs(receipt['kernel_version'])
        api.kernels_pull(receipt['kernel_version'],path=str(bundle/'remote-source'),metadata=True)
    (bundle/'execution.log').write_text(logs)
    metadata=json.loads((bundle/'remote-source/kernel-metadata.json').read_text())
    if metadata['is_private'] is not True or metadata['enable_gpu']!=receipt['gpu_requested'] or file_hash(bundle/'remote-source'/metadata['code_file'])!=receipt['source_sha256']:
        raise ValueError('Remote source/privacy/resources changed')
    reports=list((bundle/'download').rglob('stage-report.json'))
    if len(reports)!=1:raise ValueError('No unique checkpointed report; reconcile logs, never blindly resubmit')
    reportpath=reports[0];report=json.loads(reportpath.read_text());digest=file_hash(reportpath)
    if digest!=reportpath.with_suffix('.sha256').read_text().strip():raise ValueError('Report hash mismatch')
    for key in ('nonce','source_commit','training','gpu_requested'):
        if report[key]!=receipt[key]:raise ValueError('Report identity changed')
    for name,h in report['output_hashes'].items():
        p=(reportpath.parent/name).resolve()
        if not p.is_relative_to(reportpath.parent.resolve()) or p.is_symlink() or file_hash(p)!=h:raise ValueError('Artifact hash mismatch')
    write(bundle/'verification.json',{**receipt,'integrity_verified':True,'artifact_sha256':digest,'report_relative_path':str(reportpath.relative_to(bundle/'download'))})
    return report,reportpath.parent,json.loads((bundle/'verification.json').read_text())


def init(base):
    from trustretina.runner.verified_artifacts import read_stage
    registry=base/'registry.json'
    if registry.exists():raise ValueError('Registry already exists; use advance')
    configpath=ROOT/'configs/full-backbones-v5.toml';cfg=tomllib.loads(configpath.read_text())
    reuse=json.loads((base/'reuse-verification.json').read_text())
    if reuse['release_sha256']!=cfg['release_manifest_sha256']:raise ValueError('Frozen release mismatch')
    prep=ROOT/'artifacts/runner/full-v5-inputs';report,folder,_=read_stage(prep)
    if report['status']!='complete' or report['config_sha256']!=file_hash(configpath):raise ValueError('Shared cache not complete/config incompatible')
    inputs=json.loads((folder/'inputs.json').read_text())
    r={'schema':5,'config':cfg,'config_sha256':file_hash(configpath),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'experiment_source_sha256':source_hash(),'cache':{'bundle':str(prep.relative_to(ROOT)),'kernel_version':json.loads((prep/'submission.json').read_text())['kernel_version'],'inputs_sha256':file_hash(folder/'inputs.json'),'report_sha256':file_hash(folder/'stage-report.json')},'runs':[{'backbone':n,'status':'queued','attempts':[]} for n in cfg['backbones']],'selected_backbone':None,'calibration_accessed':False,'final_evaluation_accessed':False}
    write(registry,r);return r


def source_hash():
    return hashlib.sha256(b''.join(str(p.relative_to(ROOT)).encode()+p.read_bytes() for p in sorted((ROOT/'src').rglob('*.py')))).hexdigest()


def prepare(base,r,run):
    if source_hash()!=r['experiment_source_sha256']:raise ValueError('Training source changed; explicit compatibility review required')
    nonce=uuid.uuid4().hex;bundle=base/run['backbone']/f'attempt-{len(run["attempts"])+1}';bundle.mkdir(parents=True,exist_ok=False)
    spec={'config':r['config'],'config_sha256':r['config_sha256'],'inputs_sha256':r['cache']['inputs_sha256'],'experiment_source_sha256':r['experiment_source_sha256'],'backbone':run['backbone'],'dataset_sources':[],'cache_kernel_version':r['cache']['kernel_version']}
    sources=[r['cache']['kernel_version'].rsplit('/',1)[0]]
    if run['status']=='paused':
        previous=ROOT/run['attempts'][-1]['bundle'];v=json.loads((previous/'verification.json').read_text());reportpath=previous/'download'/v['report_relative_path'];progress=reportpath.parent/'progress.json'
        spec['resume']={'kernel_version':run['attempts'][-1]['kernel_version'],'progress_sha256':file_hash(progress)}
        sources.append(spec['resume']['kernel_version'].rsplit('/',1)[0])
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT/'src').rglob('*.py')):z.writestr(str(p.relative_to(ROOT)),p.read_bytes())
        z.writestr('configs/phase1.toml',(ROOT/'configs/phase1.toml').read_bytes())
    code="import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\n"+f"import base64,zipfile,io,sys,json\nfrom pathlib import Path\nz=Path('training-code');z.mkdir()\nzipfile.ZipFile(io.BytesIO(base64.b64decode({base64.b64encode(archive.getvalue()).decode()!r}))).extractall(z)\nsys.path.insert(0,str(z/'src'))\nfrom trustretina.runner.full_runtime import run\nrun(json.loads({json.dumps(spec)!r}),input_root='/kaggle/input',output='full',nonce={nonce!r},source_commit={r['source_commit']!r})\n"
    kernel='saisathwikregula/trustretina-full-v5-'+run['backbone'].replace('_','-')+'-'+nonce[:8]
    (bundle/'execute.py').write_text(code)
    write(bundle/'kernel-metadata.json',{'id':kernel,'title':kernel.split('/')[1],'code_file':'execute.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'dataset_sources':[],'competition_sources':[],'kernel_sources':sources})
    receipt={'kernel_id':kernel,'nonce':nonce,'source_commit':r['source_commit'],'source_sha256':file_hash(bundle/'execute.py'),'submission_attempted':False,'training':True,'gpu_requested':True,'execution_timeout_seconds':r['config']['maximum_gpu_session_seconds']}
    write(bundle/'submission.json',receipt)
    run['attempts'].append({'bundle':str(bundle.relative_to(ROOT)),'kernel_id':kernel,'status':'prepared'});run['status']='prepared'
    return bundle,receipt


def advance(base,r,api,submit):
    # Reconcile all known attempts before any mutation; at most one GPU job can be active.
    for run in r['runs']:
        if run['status'] not in ('running','submitting','ambiguous'):continue
        attempt=run['attempts'][-1];bundle=ROOT/attempt['bundle'];receipt=json.loads((bundle/'submission.json').read_text())
        identity=receipt.get('kernel_version',receipt['kernel_id'])
        remote=str(api.kernels_status(identity).status).split('.')[-1];attempt['remote_status']=remote
        if remote in ('QUEUED','RUNNING'):return
        report,folder,v=retrieve(api,bundle)
        run.update(status=report['status'],completed_epochs=report.get('completed_epochs'),report_sha256=v['artifact_sha256'],elapsed_seconds=report.get('elapsed_seconds'),metrics=report.get('metrics'),artifact_hashes=report['output_hashes'])
        attempt.update(status=report['status'],kernel_version=receipt['kernel_version'])
        if run['status'] not in ('complete','paused'):return
    if not submit:return
    unfinished=next((x for x in r['runs'] if x['status']!='complete'),None)
    if not unfinished:return
    if unfinished['status'] not in ('queued','paused','prepared'):return
    # Existing remote jobs are inspected before every submission. Unknown matching runs block.
    known={a['kernel_id'] for x in r['runs'] for a in x['attempts']}
    existing=[k.ref for k in api.kernels_list(mine=True,page_size=100,sort_by='dateRun') if 'trustretina-full-v5-' in k.ref and 'inputs-' not in k.ref]
    if set(existing)-known:raise ValueError('Unregistered full run exists; recover it before submitting')
    quota=api.quota_view();write(base/'latest-quota.json',json.loads(str(quota)))
    q=json.loads(str(quota))['gpuQuota']
    # Kaggle currently emits malformed fractional durations (210.693.0s); parse leading number.
    import re
    seconds=lambda s:float(re.match(r'^[0-9]+(?:\.[0-9]+)?',s).group())
    remaining=seconds(q['totalTimeAllowed'])-seconds(q['timeUsed'])-seconds(q['timeReserved'])
    r['quota_remaining_seconds_observed']=remaining
    if remaining<600:
        r['queue_blocker']='Free quota below ten-minute minimum; no paid fallback';return
    if unfinished['status']=='prepared':
        bundle=ROOT/unfinished['attempts'][-1]['bundle'];receipt=json.loads((bundle/'submission.json').read_text())
    else:bundle,receipt=prepare(base,r,unfinished)
    if receipt['submission_attempted']:raise ValueError('Ambiguous attempted submission must be reconciled; no retry')
    # Write-ahead receipt and registry BEFORE external mutation.
    receipt['submission_attempted']=True;write(bundle/'submission.json',receipt);unfinished['status']='submitting';write(base/'registry.json',r)
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            result=api.kernels_push(str(bundle),timeout=str(min(r['config']['maximum_gpu_session_seconds'],int(remaining))),acc='NvidiaTeslaT4')
        version=getattr(result,'version_number',None)
        if not version:raise ValueError('No version returned; reconcile remote identity')
        receipt.update(kernel_version_number=version,kernel_version=receipt['kernel_id']+'/'+str(version),submission_verified=True)
        write(bundle/'submission.json',receipt);unfinished['attempts'][-1].update(kernel_version=receipt['kernel_version'],status='running');unfinished['status']='running'
    except Exception as e:
        unfinished['status']='ambiguous';receipt.update(submission_error_type=type(e).__name__,submission_error=str(e)[:300]);write(bundle/'submission.json',receipt);raise


def snapshot(base, r):
    """Only aggregate state and receipts; never sample rows or probabilities."""
    bundles=[r['cache']['bundle']]+[a['bundle'] for run in r['runs'] for a in run['attempts']]
    records={b:json.loads((ROOT/b/'submission.json').read_text()) for b in bundles}
    write(ROOT/'results/full-v5/run-registry.json',{'registry':r,'receipts':records})


def restore(base):
    if (base/'registry.json').exists():raise ValueError('Private registry exists; refusing overwrite')
    saved=json.loads((ROOT/'results/full-v5/run-registry.json').read_text())
    for relative,receipt in saved['receipts'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to((ROOT/'artifacts/runner').resolve()):raise ValueError('Receipt path outside private artifacts')
        path.mkdir(parents=True,exist_ok=True)
        if (path/'submission.json').exists() and json.loads((path/'submission.json').read_text())!=receipt:raise ValueError('Existing receipt disagrees')
        write(path/'submission.json',receipt)
    write(base/'registry.json',saved['registry']);return saved['registry']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','restore','export','status','advance']);p.add_argument('--submit',action='store_true',help='Submit next authorized run only after reconciliation');p.add_argument('--base',type=Path,default=ROOT/'artifacts/runner/full-v5');args=p.parse_args();base=args.base
    if args.action=='init':r=init(base)
    elif args.action=='restore':r=restore(base)
    else:r=json.loads((base/'registry.json').read_text())
    if args.action=='advance':
        try:advance(base,r,api_client(),args.submit)
        finally:write(base/'registry.json',r)
    if args.action=='export':snapshot(base,r)
    print(json.dumps({'runs':[{k:x[k] for k in ('backbone','status','completed_epochs','elapsed_seconds') if k in x} for x in r['runs']],'selected_backbone':r['selected_backbone'],'queue_blocker':r.get('queue_blocker')},indent=2))

if __name__=='__main__':
    lockpath=ROOT/'artifacts/runner/full-v5/controller.lock'
    lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
