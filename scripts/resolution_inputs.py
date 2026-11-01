#!/usr/bin/env python3
"""One private CPU preparation job; never requests GPU or submits training."""
import argparse,base64,contextlib,io,json,subprocess,sys,time,uuid,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from trustretina.data.validation import file_hash,canonical_hash
import full_experiment as transport
BASE=ROOT/'artifacts/resolution-v13/preparation'


def prepare():
    if (BASE/'submission.json').exists():raise ValueError('Existing CPU job: inspect/retrieve, never duplicate')
    old=json.loads((ROOT/'artifacts/runner/nested-v11/registry.json').read_text());cache=old['cache']
    rows=json.loads((ROOT/cache['training_metadata_file']).read_text());nonce=uuid.uuid4().hex
    BASE.mkdir(parents=True,exist_ok=True)
    spec={'source_dataset':'andrewmvd/ocular-disease-recognition-odir5k/2','old_cache_version':cache['kernel_version'],
          'old_inputs_sha256':cache['inputs_sha256'],'rows_canonical_sha256':canonical_hash(rows),'fold_manifest_sha256':old['fold_manifest_sha256']}
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT/'src').rglob('*.py')):z.write(p,str(p.relative_to(ROOT)))
        z.write(ROOT/'scripts/resolution_prepare_runtime.py','resolution_prepare_runtime.py')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    code=f"import base64,zipfile,io,sys,json\nfrom pathlib import Path\np=Path('cpu-code');p.mkdir()\nzipfile.ZipFile(io.BytesIO(base64.b64decode({base64.b64encode(archive.getvalue()).decode()!r}))).extractall(p)\nsys.path[:0]=[str(p),str(p/'src')]\nfrom resolution_prepare_runtime import run\nrun(json.loads({json.dumps(spec)!r}),input_root='/kaggle/input',output='prepared',nonce={nonce!r},source_commit={commit!r})\n"
    (BASE/'execute.py').write_text(code)
    kernel='saisathwikregula/trustretina-res-v13-inputs-'+nonce[:8]
    transport.write(BASE/'kernel-metadata.json',{'id':kernel,'title':kernel.split('/')[1],'code_file':'execute.py','language':'python','kernel_type':'script',
          'is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':False,
          'dataset_sources':['andrewmvd/ocular-disease-recognition-odir5k/2'],'competition_sources':[],
          'kernel_sources':[cache['kernel_version'].rsplit('/',1)[0]],'docker_image':old['docker_image'],'docker_image_pinning_type':'original'})
    transport.write(BASE/'submission.json',{'kernel_id':kernel,'nonce':nonce,'source_commit':commit,'source_sha256':file_hash(BASE/'execute.py'),
          'submission_attempted':False,'training':False,'gpu_requested':False,'execution_timeout_seconds':1800,'source_dataset_version':2,'spec_sha256':canonical_hash(spec)})
    return kernel


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','submit-cpu','status','retrieve']);a=parser.parse_args()
    if a.action=='prepare':print(prepare());return
    receipt=json.loads((BASE/'submission.json').read_text());api=transport.api_client()
    if a.action=='submit-cpu':
        meta=json.loads((BASE/'kernel-metadata.json').read_text())
        if receipt['submission_attempted']:raise ValueError('Existing/ambiguous CPU attempt: recover its version, never resubmit')
        if meta['enable_gpu'] or meta['enable_tpu'] or receipt['gpu_requested'] or receipt['training'] or not meta['is_private']:
            raise ValueError('Only private nontraining CPU preparation authorized')
        if file_hash(BASE/'execute.py')!=receipt['source_sha256']:raise ValueError('CPU source changed')
        for page in range(1,21):
            jobs=list(api.kernels_list(mine=True,page=page,page_size=100,sort_by='dateRun'))
            if any('trustretina-res-v13-inputs-' in k.ref for k in jobs):raise ValueError('Remote preparation exists: recover, do not duplicate')
            if len(jobs)<100:break
        else:raise ValueError('Inventory bound exceeded')
        receipt['submission_attempted']=True;transport.write(BASE/'submission.json',receipt)
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                result=api.kernels_push(str(BASE),timeout='1800')
            if not result.version_number:raise ValueError('Missing CPU version: recover')
            receipt.update(kernel_version=receipt['kernel_id']+'/'+str(result.version_number),submission_verified=True)
            transport.write(BASE/'submission.json',receipt)
        except Exception as e:
            receipt['submission_error_type']=type(e).__name__;transport.write(BASE/'submission.json',receipt);raise
        print(json.dumps({'cpu_job':receipt['kernel_version'],'gpu_requested':False}));return
    if a.action=='status':
        if not receipt.get('kernel_version'):raise ValueError('Recover ambiguous CPU version before polling')
        print(str(api.kernels_status(receipt['kernel_version']).status));return
    report,folder,v=transport.retrieve(api,BASE)
    print(json.dumps({k:report.get(k) for k in ('status','native_images_verified','parity_mismatched_images','new_320_from_native','elapsed_seconds')}))


if __name__=='__main__':main()
