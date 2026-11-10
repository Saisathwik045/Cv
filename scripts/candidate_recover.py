#!/usr/bin/env python3
"""Restore only pinned v14 execution inputs/receipts. No preparation, approval or jobs."""
import contextlib,io,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import full_experiment as transport
from trustretina.data.validation import file_hash


def download(api,receipt,target,pattern):
    target.mkdir(parents=True,exist_ok=True);target.chmod(0o700)
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        token=None
        for _ in range(10):
            _,token=api.kernels_output(receipt['kernel_version'],path=str(target/'download'),file_pattern=pattern,page_token=token,page_size=100)
            if not token:break
        else:raise ValueError('Recovery pagination cap')
        api.kernels_pull(receipt['kernel_version'],path=str(target/'remote-source'),metadata=True)
    meta=json.loads((target/'remote-source/kernel-metadata.json').read_text())
    if not meta['is_private'] or meta['enable_gpu'] or meta['enable_tpu'] or file_hash(target/'remote-source'/meta['code_file'])!=receipt['source_sha256']:raise ValueError('Pinned CPU source/resources changed')
    reportpath=target/'download/prepared/stage-report.json';report=json.loads(reportpath.read_text())
    if file_hash(reportpath)!=reportpath.with_suffix('.sha256').read_text().strip() or report['nonce']!=receipt['nonce'] or report['source_commit']!=receipt['source_commit'] or report['status']!='complete':raise ValueError('Pinned recovery report mismatch')
    transport.write(target/'submission.json',receipt)
    return report,reportpath.parent


def main():
    import candidate_experiment as c
    saved=json.loads((ROOT/'results/candidate-v14/frozen-plan.json').read_text());p=saved['registry'];c.verify_plan(p)
    registry=c.BASE/'registry.json'
    if registry.exists():
        current=json.loads(registry.read_text());c.verify_plan(current)
        if current['plan_sha256']!=p['plan_sha256']:raise ValueError('Private/public plan differs; preserve both for reconciliation')
        p=current
        if Path(current['local']['train_folder']).joinpath('inputs.json').exists() and Path(current['local']['roles_folder']).joinpath('inputs.json').exists():
            print('Existing private inputs preserved; no download, approval or job. Use status.');return
    scratch=Path('/tmp/trustretina-v14-private/recovered');scratch.mkdir(parents=True,exist_ok=True);scratch.chmod(0o700)
    if shutil.disk_usage(scratch).free<3_300_000_000:raise ValueError('Need3.3GB free private scratch for missing actual inputs; no verified files will be deleted')
    api=transport.api_client()
    oldreceipt=json.loads((ROOT/'results/resolution-v13/preparation-receipt.json').read_text())
    trainreport,train=download(api,oldreceipt,scratch/'train',r'^prepared/(inputs.json|stage-report.json|stage-report.sha256|train-(224|320)\.npy|train-rows.json|convnext_tiny-983f1562.pth)$')
    if file_hash(train/'stage-report.json')!=p['input_report_sha256']['train']:raise ValueError('Pinned TRAIN report digest changed')
    if file_hash(train/'inputs.json')!=p['train_inputs_sha256']:raise ValueError('Wrong actual TRAIN index')
    idx=json.loads((train/'inputs.json').read_text())
    names=[idx['rows_file'],idx['weights']['convnext_tiny']['file'],*[x['file'] for x in idx['resolutions'].values()]]
    for name in names:
        if file_hash(train/name)!=trainreport['output_hashes'][name]:raise ValueError('Recovered actual TRAIN bytes differ')
    receipt=json.loads((ROOT/'results/candidate-v14/cpu-preparation-receipt.json').read_text())
    report,roles=download(api,receipt,scratch/'roles',r'^prepared/.*$')
    if file_hash(roles/'stage-report.json')!=p['input_report_sha256']['development_roles']:raise ValueError('Pinned development report digest changed')
    if file_hash(roles/'inputs.json')!=p['inputs_sha256']:raise ValueError('Wrong actual development role index')
    for name,digest in report['output_hashes'].items():
        if file_hash(roles/name)!=digest:raise ValueError('Recovered role bytes differ')
    p['local']={'train_folder':str(train),'roles_folder':str(roles),'preparation_bundle':str(scratch/'roles')}
    c.BASE.mkdir(parents=True,exist_ok=True)
    if registry.exists():
        prior=json.loads(registry.read_text())
        if prior['plan_sha256']!=p['plan_sha256']:raise ValueError('Preserve newer registry: plan identity differs')
        prior['local']=p['local'];p=prior
    transport.write(registry,p)
    for relative,receipt in saved['receipts'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(c.BASE.resolve()):raise ValueError('Receipt outside execution directory')
        path.mkdir(parents=True,exist_ok=True)
        if not (path/'submission.json').exists():transport.write(path/'submission.json',receipt)
    print('Pinned actual TRAIN224/TRAIN320 and development224/320 inputs restored. No jobs or approval created.')


if __name__=='__main__':main()
