#!/usr/bin/env python3
"""One authorized held-out evaluation; immutable parameters and write-ahead stage receipts."""
import argparse,base64,contextlib,fcntl,io,json,subprocess,sys,tomllib,uuid,zipfile,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import full_experiment as transport
from trustretina.data.validation import file_hash
from trustretina.runner.verified_artifacts import read_stage
BASE=ROOT/'artifacts/runner/phase7'
write=transport.write


def init():
    if (BASE/'registry.json').exists():raise ValueError('Phase7 exists; recover, never repeat initialization')
    cfgpath=ROOT/'configs/final-evaluation-v7.toml';doc=ROOT/'docs/FINAL_EVALUATION_V7.md';cfg=tomllib.loads(cfgpath.read_text())
    subprocess.run(['git','diff','--exit-code','HEAD','--',str(cfgpath.relative_to(ROOT)),str(doc.relative_to(ROOT))],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    protocol_commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(cfgpath.relative_to(ROOT))],cwd=ROOT,text=True).strip()
    contractpath=ROOT/'results/reliability-v6/candidate-contract.json'
    if file_hash(contractpath)!=cfg['candidate_contract_sha256']:raise ValueError('Frozen contract changed')
    contract=json.loads(contractpath.read_text());p6=json.loads((ROOT/'artifacts/runner/phase6/registry.json').read_text())
    old,oldfolder,oldv=read_stage(ROOT/p6['stages']['analysis']['bundle']);gpu,_,_=read_stage(ROOT/p6['stages']['inference']['bundle'])
    if old['status']!='complete' or file_hash(oldfolder/'temperature.json')!=contract['temperature_sha256'] or file_hash(oldfolder/'quality-normalizer.npz')!=contract['quality_normalizer_sha256'] or file_hash(oldfolder/'candidate-pipeline.json')!=contract['artifact_storage']['candidate_sha256']:raise ValueError('Frozen Phase6 artifact mismatch')
    temperature=json.loads((oldfolder/'temperature.json').read_text())['temperature']
    if temperature!=contract['temperature'] or contract['checkpoint_sha256']!=cfg['checkpoint_sha256']:raise ValueError('Candidate parameter identity mismatch')
    checkpoint=ROOT/'artifacts/runner/full-v5/convnext_tiny/attempt-1/download/full/best.pt'
    if file_hash(checkpoint)!=cfg['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
    release=ROOT/'artifacts/runner/contained-release-v4/frozen-splits';complete=json.loads((release/'complete.json').read_text());checksums=json.loads((release/'checksums.json').read_text())
    if file_hash(release/'manifest.json')!=cfg['release_sha256'] or complete['manifest_sha256']!=cfg['release_sha256'] or file_hash(release/'checksums.json')!=complete['checksums_sha256']:raise ValueError('Frozen split changed')
    manifest=json.loads((release/'manifest.json').read_text());BASE.mkdir(parents=True,exist_ok=True)
    # Append before the first final-label read, after the protocol is committed.
    exposure={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':'authorized_first_final_manifest_open','expected_sha256':checksums['final_evaluation.json'],'protocol_commit':protocol_commit,'protocol_sha256':file_hash(doc),'purpose':'single frozen held-out research evaluation; no fitting'}
    with (BASE/'exposure.jsonl').open('a') as f:
        import os
        f.write(json.dumps(exposure)+'\n');f.flush();os.fsync(f.fileno())
    final=release/'final_evaluation.json'
    if file_hash(final)!=checksums[final.name]:raise ValueError('Final role bytes changed')
    content=json.loads(final.read_text());rows=content['rows']
    if content['role']!='final_evaluation' or len(rows)!=2*cfg['expected_patients'] or {r['patient_id'] for r in rows}!=set(manifest['partitions']['final_evaluation']):raise ValueError('Final coverage mismatch')
    patients={}
    for row in rows:
        pair=patients.setdefault(row['patient_id'],{})
        if row['eye'] in pair:raise ValueError('Repeated eye')
        pair[row['eye']]=row
    if len(patients)!=cfg['expected_patients'] or any(set(v)!={'left','right'} or v['left']['labels']!=v['right']['labels'] or v['left']['component_id']!=v['right']['component_id'] for v in patients.values()):raise ValueError('Both eyes/labels/components required')
    for role in ('train','validation','calibration'):
        path=release/(role+'.json')
        if file_hash(path)!=checksums[path.name]:raise ValueError('Historical role changed')
        previous=json.loads(path.read_text())['rows']
        for key in ('patient_id','component_id'):
            if {r[key] for r in rows}&{r[key] for r in previous}:raise ValueError('Final/development separation violated')
    r={'schema':7,'config':cfg,'config_sha256':file_hash(cfgpath),'protocol_sha256':file_hash(doc),'protocol_commit_before_final_access':protocol_commit,'contract':contract,'candidate_contract_sha256':cfg['candidate_contract_sha256'],'checkpoint_sha256':cfg['checkpoint_sha256'],'release_sha256':cfg['release_sha256'],'final_manifest_sha256':checksums[final.name],'final_patients':len(patients),'final_components':len({r['component_id'] for r in rows}),'phase6_analysis_kernel_version':p6['stages']['analysis']['kernel_version'],'phase6_analysis_report_sha256':oldv['artifact_sha256'],'development_comparisons_sha256':old['output_hashes']['development-comparisons.json'],'selected_kernel_version':contract['selected_checkpoint_kernel_version'],'selected_provenance':p6['selected_provenance'],'expected_gpu_dependencies':gpu['dependencies'],'docker_image':json.loads((ROOT/p6['stages']['inference']['bundle']/'remote-source/kernel-metadata.json').read_text())['docker_image'],'stages':{},'final_data_exposure_recorded':True,'parameter_fitting':False,'backbone_training':False,'risk_policy_fitting':False,'deployment_default':'REVIEW REQUIRED'}
    spec={'rows':rows,'role':'final_evaluation','manifest_sha256':r['final_manifest_sha256'],'protocol_sha256':r['protocol_sha256'],'protocol_commit':protocol_commit,'release_sha256':r['release_sha256'],'checkpoint_sha256':r['checkpoint_sha256'],'candidate_contract_sha256':r['candidate_contract_sha256'],'maximum_seconds':cfg['maximum_cpu_input_seconds']}
    write(BASE/'input-spec.json',spec);write(BASE/'registry.json',r);return r


def build(r,stage):
    if stage in r['stages']:raise ValueError('Stage already prepared/attempted; preserve and reconcile, no replacement')
    cfg=r['config'];gpu=stage=='inference';nonce=uuid.uuid4().hex;bundle=BASE/stage/'attempt-1';bundle.mkdir(parents=True,exist_ok=False)
    spec={'config':cfg,'config_sha256':r['config_sha256'],'protocol_sha256':r['protocol_sha256'],'candidate_contract_sha256':r['candidate_contract_sha256'],'contract':r['contract'],'checkpoint_sha256':r['checkpoint_sha256'],'release_sha256':r['release_sha256'],'final_manifest_sha256':r['final_manifest_sha256']}
    if stage=='inputs':
        spec=json.loads((BASE/'input-spec.json').read_text());module='final_inputs';sources=[r['selected_kernel_version'].rsplit('/',1)[0]];datasets=['andrewmvd/ocular-disease-recognition-odir5k'];timeout=cfg['maximum_cpu_input_seconds'];output='prepared'
    elif stage=='inference':
        if r['stages'].get('inputs',{}).get('status')!='complete':raise ValueError('Verified input stage required')
        spec.update(inputs_sha256=r['stages']['inputs']['inputs_sha256'],selected_provenance=r['selected_provenance'],expected_dependencies=r['expected_gpu_dependencies'])
        module='final_runtime';sources=[r['stages']['inputs']['kernel_version'].rsplit('/',1)[0]];datasets=[];timeout=cfg['maximum_gpu_seconds'];output='full'
    elif stage=='analysis':
        if r['stages'].get('inference',{}).get('status')!='complete':raise ValueError('Verified inference required')
        spec.update(inputs_sha256=r['stages']['inputs']['inputs_sha256'],inference_report_sha256=r['stages']['inference']['report_sha256'],phase6_analysis_report_sha256=r['phase6_analysis_report_sha256'],development_comparisons_sha256=r['development_comparisons_sha256'])
        module='final_analysis';sources=[r['stages'][k]['kernel_version'].rsplit('/',1)[0] for k in ('inputs','inference')]+[r['phase6_analysis_kernel_version'].rsplit('/',1)[0]];datasets=[];timeout=cfg['maximum_cpu_analysis_seconds'];output='full'
    else:raise ValueError('Unknown stage')
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted((ROOT/'src').rglob('*.py')):z.writestr(str(path.relative_to(ROOT)),path.read_bytes())
        z.writestr('configs/phase1.toml',(ROOT/'configs/phase1.toml').read_bytes())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    code="import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\n"+f"import base64,zipfile,io,sys,json\nfrom pathlib import Path\nz=Path('evaluation-code');z.mkdir()\nzipfile.ZipFile(io.BytesIO(base64.b64decode({base64.b64encode(archive.getvalue()).decode()!r}))).extractall(z)\nsys.path.insert(0,str(z/'src'))\nfrom trustretina.runner.{module} import run\nrun(json.loads({json.dumps(spec)!r}),input_root='/kaggle/input',output={output!r},nonce={nonce!r},source_commit={commit!r})\n"
    (bundle/'execute.py').write_text(code);kernel='saisathwikregula/trustretina-phase7-'+stage+'-'+nonce[:8]
    metadata={'id':kernel,'title':kernel.split('/')[1],'code_file':'execute.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':gpu,'enable_tpu':False,'enable_internet':False,'dataset_sources':datasets,'competition_sources':[],'kernel_sources':sources}
    if gpu:metadata.update(docker_image=r['docker_image'],docker_image_pinning_type='original')
    write(bundle/'kernel-metadata.json',metadata)
    receipt={'kernel_id':kernel,'nonce':nonce,'source_commit':commit,'source_sha256':file_hash(bundle/'execute.py'),'source_tree_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'src').rglob('*.py'))},'submission_attempted':False,'training':False,'gpu_requested':gpu,'execution_timeout_seconds':timeout,'protocol_sha256':r['protocol_sha256'],'protocol_commit_before_final_access':r['protocol_commit_before_final_access']}
    write(bundle/'submission.json',receipt);r['stages'][stage]={'status':'prepared','bundle':str(bundle.relative_to(ROOT)),'kernel_id':kernel,'attempt':1};write(BASE/'registry.json',r)


def advance(r,stage,submit):
    api=transport.api_client()
    if stage not in r['stages']:
        if not submit:return
        build(r,stage)
    state=r['stages'][stage];bundle=ROOT/state['bundle'];receipt=json.loads((bundle/'submission.json').read_text())
    if state['status']=='complete':return
    if receipt['submission_attempted']:
        if not receipt.get('kernel_version'):raise ValueError('Ambiguous attempted job; reconcile exact ID/source before any correction')
        status=str(api.kernels_status(receipt['kernel_version']).status).split('.')[-1];state['remote_status']=status
        if status in ('RUNNING','QUEUED'):return
        report,folder,verified=transport.retrieve(api,bundle)
        if report['protocol_sha256']!=r['protocol_sha256'] or report['candidate_contract_sha256']!=r['candidate_contract_sha256'] or report.get('parameter_fitting') is not False:raise ValueError('Frozen evaluation scope mismatch')
        state.update(status=report['status'],report_sha256=verified['artifact_sha256'],kernel_version=receipt['kernel_version'],artifact_hashes=report['output_hashes'],elapsed_seconds=report['elapsed_seconds'])
        if 'inputs_sha256' in report:state['inputs_sha256']=report['inputs_sha256']
        if report['status']!='complete':state['error']=report.get('error');raise RuntimeError('Preserved failed attempt; document defect/correction/exposure before any rerun: '+str(state['error']))
    elif submit:
        known={s['kernel_id'] for s in r['stages'].values()};remote=[k.ref for k in api.kernels_list(mine=True,page_size=100,sort_by='dateRun') if 'trustretina-phase7-' in k.ref]
        if set(remote)-known:raise ValueError('Unregistered final evaluation job exists; recover, never duplicate')
        quota=json.loads(str(api.quota_view()));write(BASE/(stage+'-quota-before.json'),quota)
        if receipt['gpu_requested']:
            import re
            seconds=lambda x:float(re.match(r'^[0-9]+(?:\.[0-9]+)?',x).group());q=quota['gpuQuota']
            if q['isPayToScaleEnabled'] is not False or seconds(q['totalTimeAllowed'])-seconds(q['timeUsed'])-seconds(q['timeReserved'])<receipt['execution_timeout_seconds']:raise ValueError('Insufficient free quota or paid scaling enabled; no fallback')
        receipt['submission_attempted']=True;write(bundle/'submission.json',receipt);state['status']='submitting';write(BASE/'registry.json',r)
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):result=api.kernels_push(str(bundle),timeout=str(receipt['execution_timeout_seconds']),acc='NvidiaTeslaT4' if receipt['gpu_requested'] else None)
            if not result.version_number:raise ValueError('No version returned; reconcile')
            receipt.update(kernel_version_number=result.version_number,kernel_version=receipt['kernel_id']+'/'+str(result.version_number),submission_verified=True);write(bundle/'submission.json',receipt);state.update(status='running',kernel_version=receipt['kernel_version'])
        except Exception as error:state['status']='ambiguous';receipt['submission_error_type']=type(error).__name__;write(bundle/'submission.json',receipt);raise


def export(r):
    write(ROOT/'results/final-v7/run-registry.json',{'registry':r,'receipts':{key:json.loads((ROOT/s['bundle']/'submission.json').read_text()) for key,s in r['stages'].items()}})


def restore():
    if (BASE/'registry.json').exists():raise ValueError('Private final evaluation state exists; refusing overwrite')
    saved=json.loads((ROOT/'results/final-v7/run-registry.json').read_text());r=saved['registry']
    for key,receipt in saved['receipts'].items():
        folder=(ROOT/r['stages'][key]['bundle']).resolve()
        if not folder.is_relative_to(BASE.resolve()):raise ValueError('Receipt path outside private storage')
        write(folder/'submission.json',receipt)
    write(BASE/'registry.json',r);return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','restore','status','advance','retrieve','export']);p.add_argument('--stage',choices=['inputs','inference','analysis']);p.add_argument('--submit',action='store_true');a=p.parse_args()
    if a.action=='init':r=init()
    elif a.action=='restore':r=restore()
    else:r=json.loads((BASE/'registry.json').read_text())
    if a.action=='advance':
        if not a.stage:raise ValueError('--stage required')
        try:advance(r,a.stage,a.submit)
        finally:write(BASE/'registry.json',r)
    if a.action=='retrieve':
        api=transport.api_client()
        for s in r['stages'].values():
            if s['status']=='complete':transport.retrieve(api,ROOT/s['bundle'])
    if a.action=='export':export(r)
    print(json.dumps({'stages':r['stages'],'parameter_fitting':False},indent=2))

if __name__=='__main__':
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
