#!/usr/bin/env python3
"""Three bounded private Kaggle stages; persist/reconcile receipts before submission."""
import argparse,base64,contextlib,hashlib,io,json,os,subprocess,sys,tomllib,uuid,zipfile,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import full_experiment as transport
from trustretina.data.validation import file_hash
from trustretina.runner.verified_artifacts import read_stage
BASE=ROOT/'artifacts/runner/phase6'


def write(path,value):transport.write(path,value)


def init():
    if (BASE/'registry.json').exists():raise ValueError('Phase6 state exists; inspect/advance, never reinitialize')
    config=ROOT/'configs/reliability-v6.toml';cfg=tomllib.loads(config.read_text());protocol=ROOT/'docs/RELIABILITY_V6.md'
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(config.relative_to(ROOT))],cwd=ROOT,text=True).strip()
    # Protocol must already be committed before reading calibration rows/labels.
    subprocess.run(['git','diff','--exit-code','HEAD','--',str(config.relative_to(ROOT)),str(protocol.relative_to(ROOT))],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    release=ROOT/'artifacts/runner/contained-release-v4/frozen-splits';complete=json.loads((release/'complete.json').read_text());index=json.loads((release/'checksums.json').read_text())
    if file_hash(release/'manifest.json')!=cfg['release_sha256'] or complete['manifest_sha256']!=cfg['release_sha256'] or file_hash(release/'checksums.json')!=complete['checksums_sha256']:raise ValueError('Frozen release changed')
    readiness=json.loads((release/'readiness.json').read_text())
    if file_hash(release/'readiness.json')!=index['readiness.json'] or not readiness['checks_passed']:raise ValueError('Frozen readiness changed')
    selected=ROOT/'artifacts/runner/full-v5/convnext_tiny/attempt-1';report,folder,verification=read_stage(selected)
    if report['status']!='complete' or file_hash(folder/'best.pt')!=cfg['checkpoint_sha256'] or report['selection']['best_epoch']!=1:raise ValueError('Selected checkpoint changed')
    roles={}
    for role in ('train','validation','calibration'):
        p=release/(role+'.json')
        if file_hash(p)!=index[p.name]:raise ValueError('Frozen permitted role changed')
        content=json.loads(p.read_text())
        if content['role']!=role:raise ValueError('Role mismatch')
        roles[role]={'manifest_sha256':index[p.name],'rows':content['rows']}
    for key in ('patient_id','component_id'):
        groups=[{row[key] for row in roles[role]['rows']} for role in roles]
        if any(groups[i]&groups[j] for i in range(3) for j in range(i)):raise ValueError('Permitted-role separation changed')
    old=ROOT/'artifacts/runner/full-v5-inputs';cache,_,_=read_stage(old)
    r={'schema':6,'config':cfg,'config_sha256':file_hash(config),'protocol_sha256':file_hash(protocol),'protocol_commit':commit,'release_sha256':cfg['release_sha256'],'checkpoint_sha256':cfg['checkpoint_sha256'],'selected_kernel_version':json.loads((selected/'submission.json').read_text())['kernel_version'],'selected_report_sha256':verification['artifact_sha256'],'selected_provenance':report['provenance'],'expected_gpu_dependencies':report['dependencies'],'docker_image':json.loads((selected/'remote-source/kernel-metadata.json').read_text())['docker_image'],'previous_inputs_sha256':cache['inputs_sha256'],'previous_input_kernel':json.loads((old/'submission.json').read_text())['kernel_version'],'roles':{k:{'manifest_sha256':v['manifest_sha256'],'images':len(v['rows'])} for k,v in roles.items()},'stages':{},'final_evaluation_accessed':False,'backbone_training':False,'risk_policy_fitted':False}
    spec={'roles':{k:roles[k] for k in ('validation','calibration')},'protocol_sha256':r['protocol_sha256'],'protocol_commit':commit,'release_sha256':r['release_sha256'],'previous_inputs_sha256':r['previous_inputs_sha256'],'maximum_seconds':cfg['maximum_cpu_input_seconds']}
    BASE.mkdir(exist_ok=True,parents=True);write(BASE/'input-spec.json',spec);write(BASE/'registry.json',r);return r


def build(r,stage):
    if stage in r['stages']:raise ValueError('Stage already prepared; inspect existing receipt')
    cfg=r['config'];gpu=stage=='inference';nonce=uuid.uuid4().hex;bundle=BASE/stage;bundle.mkdir(exist_ok=False)
    spec={'config':cfg,'config_sha256':r['config_sha256'],'protocol_sha256':r['protocol_sha256'],'release_sha256':r['release_sha256'],'checkpoint_sha256':r['checkpoint_sha256']}
    if stage=='inputs':
        spec=json.loads((BASE/'input-spec.json').read_text());module='reliability_inputs';function='prepare';sources=[r['previous_input_kernel'].rsplit('/',1)[0]];datasets=['andrewmvd/ocular-disease-recognition-odir5k'];timeout=cfg['maximum_cpu_input_seconds'];output='prepared'
    elif stage=='inference':
        if r['stages'].get('inputs',{}).get('status')!='complete':raise ValueError('Verified inputs required')
        spec.update(inputs_sha256=r['stages']['inputs']['inputs_sha256'],selected_provenance=r['selected_provenance'],expected_dependencies=r['expected_gpu_dependencies'])
        module='reliability_runtime';function='run';sources=[r['stages']['inputs']['kernel_version'].rsplit('/',1)[0],r['selected_kernel_version'].rsplit('/',1)[0]];datasets=[];timeout=cfg['maximum_gpu_seconds'];output='full'
    elif stage=='analysis':
        if r['stages'].get('inference',{}).get('status')!='complete':raise ValueError('Verified inference required')
        spec.update(inputs_sha256=r['stages']['inputs']['inputs_sha256'],inference_report_sha256=r['stages']['inference']['report_sha256'])
        module='reliability_analysis';function='run';sources=[r['stages'][k]['kernel_version'].rsplit('/',1)[0] for k in ('inputs','inference')];datasets=[];timeout=cfg['maximum_cpu_analysis_seconds'];output='full'
    else:raise ValueError('Unknown stage')
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((ROOT/'src').rglob('*.py')):archive.writestr(str(p.relative_to(ROOT)),p.read_bytes())
        archive.writestr('configs/phase1.toml',(ROOT/'configs/phase1.toml').read_bytes())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    code="import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\n"+f"import base64,zipfile,io,sys,json\nfrom pathlib import Path\nz=Path('phase6-code');z.mkdir()\nzipfile.ZipFile(io.BytesIO(base64.b64decode({base64.b64encode(buffer.getvalue()).decode()!r}))).extractall(z)\nsys.path.insert(0,str(z/'src'))\nfrom trustretina.runner.{module} import {function}\n{function}(json.loads({json.dumps(spec)!r}),input_root='/kaggle/input',output={output!r},nonce={nonce!r},source_commit={commit!r})\n"
    (bundle/'execute.py').write_text(code);kernel='saisathwikregula/trustretina-phase6-'+stage+'-'+nonce[:8]
    meta={'id':kernel,'title':kernel.split('/')[1],'code_file':'execute.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':gpu,'enable_tpu':False,'enable_internet':False,'dataset_sources':datasets,'competition_sources':[],'kernel_sources':sources}
    if gpu:meta['docker_image']=r['docker_image'];meta['docker_image_pinning_type']='original'
    write(bundle/'kernel-metadata.json',meta)
    receipt={'kernel_id':kernel,'nonce':nonce,'source_commit':commit,'source_sha256':file_hash(bundle/'execute.py'),'source_tree_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'src').rglob('*.py'))},'submission_attempted':False,'training':False,'gpu_requested':gpu,'execution_timeout_seconds':timeout,'protocol_sha256':r['protocol_sha256'],'protocol_commit':r['protocol_commit']}
    write(bundle/'submission.json',receipt);r['stages'][stage]={'status':'prepared','bundle':str(bundle.relative_to(ROOT)),'kernel_id':kernel};write(BASE/'registry.json',r)


def advance(r,stage,submit):
    api=transport.api_client()
    if stage not in r['stages']:
        if not submit:return
        build(r,stage)
    state=r['stages'][stage];bundle=ROOT/state['bundle'];receipt=json.loads((bundle/'submission.json').read_text())
    if state['status']=='complete':return
    if receipt['submission_attempted']:
        if not receipt.get('kernel_version'):raise ValueError('Attempted submission ambiguous; reconcile remote ID/source, never blindly resubmit')
        status=str(api.kernels_status(receipt['kernel_version']).status).split('.')[-1];state['remote_status']=status
        if status in ('RUNNING','QUEUED'):return
        report,folder,v=transport.retrieve(api,bundle)
        if report['protocol_sha256']!=r['protocol_sha256'] or report.get('final_evaluation_access') is not False:raise ValueError('Phase6 role/protocol mismatch')
        state.update(status=report['status'],report_sha256=v['artifact_sha256'],kernel_version=receipt['kernel_version'],artifact_hashes=report['output_hashes'],elapsed_seconds=report.get('elapsed_seconds'))
        if 'inputs_sha256' in report:state['inputs_sha256']=report['inputs_sha256']
        if report['status']!='complete':state['error']=report.get('error');return
    elif submit:
        known={s['kernel_id'] for s in r['stages'].values()}
        remote=[k.ref for k in api.kernels_list(mine=True,page_size=100,sort_by='dateRun') if 'trustretina-phase6-' in k.ref]
        if set(remote)-known:raise ValueError('Unregistered Phase6 job exists; reconcile before submission')
        quota=json.loads(str(api.quota_view()));write(BASE/(stage+'-quota-before.json'),quota)
        if receipt['gpu_requested']:
            import re
            seconds=lambda s:float(re.match(r'^[0-9]+(?:\.[0-9]+)?',s).group())
            q=quota['gpuQuota'];available=seconds(q['totalTimeAllowed'])-seconds(q['timeUsed'])-seconds(q['timeReserved'])
            if q['isPayToScaleEnabled'] is not False or available<receipt['execution_timeout_seconds']:raise ValueError('Insufficient free quota or paid scaling enabled; save queue, no fallback')
        receipt['submission_attempted']=True;write(bundle/'submission.json',receipt);state['status']='submitting';write(BASE/'registry.json',r)
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):response=api.kernels_push(str(bundle),timeout=str(receipt['execution_timeout_seconds']),acc='NvidiaTeslaT4' if receipt['gpu_requested'] else None)
            if not response.version_number:raise ValueError('No version returned; reconcile')
            receipt.update(kernel_version_number=response.version_number,kernel_version=receipt['kernel_id']+'/'+str(response.version_number),submission_verified=True)
            state.update(status='running',kernel_version=receipt['kernel_version']);write(bundle/'submission.json',receipt)
        except Exception as e:
            state['status']='ambiguous';receipt['submission_error_type']=type(e).__name__;write(bundle/'submission.json',receipt);raise


def export(r):
    receipts={key:json.loads((ROOT/s['bundle']/'submission.json').read_text()) for key,s in r['stages'].items()}
    write(ROOT/'results/reliability-v6/run-registry.json',{'registry':r,'receipts':receipts})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','status','advance','export']);p.add_argument('--stage',choices=['inputs','inference','analysis']);p.add_argument('--submit',action='store_true');a=p.parse_args()
    if a.action=='init':r=init()
    else:r=json.loads((BASE/'registry.json').read_text())
    if a.action=='advance':
        if not a.stage:raise ValueError('--stage required')
        try:advance(r,a.stage,a.submit)
        finally:write(BASE/'registry.json',r)
    if a.action=='export':export(r)
    print(json.dumps({'stages':r['stages'],'final_evaluation_accessed':r['final_evaluation_accessed']},indent=2))

if __name__=='__main__':
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
