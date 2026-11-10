#!/usr/bin/env python3
"""Authorized one-session CPU readiness controller; never submits a GPU job."""
import argparse,base64,contextlib,io,json,subprocess,sys,uuid,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import full_experiment as transport
from trustretina.data.validation import file_hash,canonical_hash
BASE=ROOT/'artifacts/candidate-v14/preparation'


def prepare():
    if (BASE/'submission.json').exists():raise ValueError('Existing CPU receipt: recover, never prepare/submit again')
    recovery=json.loads((ROOT/'artifacts/candidate-v14/input-recovery.json').read_text());old=json.loads((ROOT/'results/resolution-v13/queue.json').read_text())['registry'];plan=json.loads((ROOT/'results/candidate-v14/proposed-plan.json').read_text())
    sources={'train':{'kernel_version':old['cache']['kernel_version'],'inputs_sha256':old['cache']['inputs_sha256']}}
    for role,key in [('validation','v5'),('calibration','v6')]:sources[role]={k:recovery['roles'][key][k] for k in ('kernel_version','inputs_sha256')}
    spec={'source_dataset':'andrewmvd/ocular-disease-recognition-odir5k/2','release_sha256':plan['frozen_v4_release_sha256'],'sources':sources,'manifest_sha256':{k:v['manifest_sha256'] for k,v in plan['partitions'].items()},'maximum_seconds':3600}
    BASE.mkdir(parents=True);nonce=uuid.uuid4().hex;commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted((ROOT/'src').rglob('*.py')):z.write(path,str(path.relative_to(ROOT)))
        z.write(ROOT/'scripts/candidate_prepare_runtime.py','candidate_prepare_runtime.py')
    code=f"import base64,zipfile,io,sys,json\nfrom pathlib import Path\np=Path('cpu-code');p.mkdir()\nzipfile.ZipFile(io.BytesIO(base64.b64decode({base64.b64encode(archive.getvalue()).decode()!r}))).extractall(p)\nsys.path[:0]=[str(p),str(p/'src')]\nfrom candidate_prepare_runtime import run\nrun(json.loads({json.dumps(spec)!r}),input_root='/kaggle/input',output='prepared',nonce={nonce!r},source_commit={commit!r})\n"
    (BASE/'execute.py').write_text(code);kernel='saisathwikregula/trustretina-v14-inputs-'+nonce[:8]
    transport.write(BASE/'kernel-metadata.json',{'id':kernel,'title':kernel.split('/')[1],'code_file':'execute.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':False,'dataset_sources':[spec['source_dataset']],'competition_sources':[],'kernel_sources':[s['kernel_version'].rsplit('/',1)[0] for s in sources.values()],'docker_image':old['docker_image'],'docker_image_pinning_type':'original'})
    transport.write(BASE/'submission.json',{'kernel_id':kernel,'nonce':nonce,'source_commit':commit,'source_sha256':file_hash(BASE/'execute.py'),'training':False,'gpu_requested':False,'submission_attempted':False,'execution_timeout_seconds':3600,'spec_sha256':canonical_hash(spec),'source_dataset_version':2})
    return kernel


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('action',choices=['prepare','submit-cpu','status','retrieve']);args=a.parse_args()
    if args.action=='prepare':print(prepare());return
    receipt=json.loads((BASE/'submission.json').read_text());api=transport.api_client()
    if args.action=='submit-cpu':
        meta=json.loads((BASE/'kernel-metadata.json').read_text())
        if receipt['submission_attempted'] or meta['enable_gpu'] or meta['enable_tpu'] or receipt['training'] or receipt['gpu_requested'] or not meta['is_private'] or receipt['execution_timeout_seconds']!=3600:raise ValueError('One private CPU-only session permitted')
        if file_hash(BASE/'execute.py')!=receipt['source_sha256']:raise ValueError('CPU source changed')
        for page in range(1,21):
            jobs=list(api.kernels_list(mine=True,page=page,page_size=100,sort_by='dateRun'))
            if any('trustretina-v14-inputs-' in j.ref for j in jobs):raise ValueError('Remote v14 CPU job exists: reconcile it; no replacement')
            if len(jobs)<100:break
        else:raise ValueError('Remote inventory bound')
        receipt['submission_attempted']=True;transport.write(BASE/'submission.json',receipt)
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):result=api.kernels_push(str(BASE),timeout='3600')
            if not result.version_number:raise ValueError('Ambiguous CPU push: preserve receipt, never resubmit')
            receipt.update(kernel_version=receipt['kernel_id']+'/'+str(result.version_number),submission_verified=True);transport.write(BASE/'submission.json',receipt)
        except Exception as e:
            receipt['error_type']=type(e).__name__;transport.write(BASE/'submission.json',receipt);raise
        print(json.dumps({'CPU_job':receipt['kernel_version'],'GPU':False}));return
    if not receipt.get('kernel_version'):raise ValueError('Resolve ambiguous CPU version before status/retrieve')
    if args.action=='status':print(str(api.kernels_status(receipt['kernel_version']).status));return
    # New tensors live in free private scratch; preserve the small receipt/source in workspace.
    target=Path('/tmp/trustretina-v14-private/preparation');target.mkdir(parents=True,exist_ok=True);target.chmod(0o700)
    import shutil
    if not (target/'submission.json').exists():shutil.copyfile(BASE/'submission.json',target/'submission.json')
    report,folder,verification=transport.retrieve(api,target)
    transport.write(BASE/'recovery.json',{'bundle':str(target),'folder':str(folder),'report_sha256':verification['artifact_sha256'],'kernel_version':receipt['kernel_version']})
    print(json.dumps({k:report.get(k) for k in ('status','native_images_verified','elapsed_seconds','calibration_labels_in_output')}))


if __name__=='__main__':main()
