#!/usr/bin/env python3
"""Read-only v13 recovery check for v14 planning. Never prepares inputs or submits jobs."""
import argparse
import contextlib
import datetime
import io
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trustretina.data.validation import file_hash
from trustretina.runner.verified_artifacts import read_stage
import resolution_experiment as c


def verify_listing(files,report):
    listing={f.file_name:f for f in files}
    required={f'full/{n}' for n in report['output_hashes']}|{'full/stage-report.json','full/stage-report.sha256'}
    if required-set(listing):raise ValueError('Pinned remote output files missing')
    return listing,len(required)


def session_outputs(api,version):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    owner,slug,label=api.parse_kernel_string(version);files=[];token=None
    with api.build_kaggle_client() as client:
        for _ in range(5):
            req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=slug
            req.version_label=api._kernel_version_label(label);req.page_size=100;req.page_token=token
            response=client.kernels.kernels_api_client.list_kernel_session_output(req)
            files.extend(response.files);token=response.next_page_token
            if not token:return files
    raise ValueError('Remote session-output inventory bound exceeded')


def verify_checkpoint_range(url,expected_size):
    import requests
    # Do not log signed URLs. At most one body byte is consumed, even if Range is ignored.
    try:
        with requests.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=45) as response:
            if response.status_code not in (200,206):raise ValueError('Remote checkpoint range request failed')
            size=int(response.headers['Content-Range'].split('/')[-1]) if response.status_code==206 else int(response.headers['Content-Length'])
            if size!=expected_size or len(response.raw.read(1))!=1:raise ValueError('Remote checkpoint length/byte retrieval differs')
    except requests.RequestException as e:
        raise ValueError('Remote checkpoint access failed; signed URL omitted') from None


def main(remote):
    out=ROOT/'results/candidate-v14';out.mkdir(exist_ok=True)
    private=ROOT/'artifacts/candidate-v14';private.mkdir(exist_ok=True);private.chmod(0o700)
    old=ROOT/'results/resolution-v13'
    for name,digest in json.loads((old/'aggregate-checksums.json').read_text()).items():
        if file_hash(old/name)!=digest:raise ValueError('Committed v13 aggregate checksum differs')
    queue=json.loads((old/'queue.json').read_text());p=queue['registry']
    c.verify_plan(p)
    if len(p['runs'])!=6 or any(r['status']!='complete' or len(r['attempts'])!=1 for r in p['runs']):raise ValueError('Expected six completed single-attempt fits')
    api=c.transport.api_client() if remote else None
    result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_checked':remote,
            'prior_plan_sha256':p['plan_sha256'],'new_jobs_submitted':0,'fitting_performed':False,
            'verification_scope':'Existing receipt/source/report/output hashes locally; remote completed-version inventory plus fresh report bytes. Large checkpoints checked locally in full; one remote byte and total length checked per checkpoint, not full redownload.', 'runs':[]}
    for run in p['runs']:
        bundle=ROOT/run['attempts'][0]['bundle'];report,folder,v=read_stage(bundle)
        if not report.get('strict_best_reload') or report['status']!='complete':raise ValueError('Uncompleted or non-strict fit')
        row={'run_id':run['id'],'kernel_version':v['kernel_version'],'local_all_outputs_sha256_verified':True,
             'report_sha256':v['artifact_sha256'],'best_sha256':report['output_hashes']['best.pt'],'last_sha256':report['output_hashes']['last.pt']}
        if remote:
            status=str(api.kernels_status(v['kernel_version']).status).split('.')[-1]
            if status!='COMPLETE':raise ValueError('Pinned remote version is not complete')
            listing,count=verify_listing(session_outputs(api,v['kernel_version']),report)
            for name in ('best.pt','last.pt'):
                verify_checkpoint_range(listing['full/'+name].url,(folder/name).stat().st_size)
            target=private/'fresh-reports'/run['id']
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                _,token=api.kernels_output(v['kernel_version'],path=str(target),file_pattern=r'^full/stage-report\.json$',page_size=100)
            if token:raise ValueError('Unexpected report-download pagination')
            if file_hash(target/'full/stage-report.json')!=v['artifact_sha256']:raise ValueError('Fresh remote report hash differs')
            row.update(remote_status=status,remote_output_inventory_verified=count,fresh_remote_report_sha256_verified=True,remote_checkpoint_range_access_verified=True)
        result['runs'].append(row)
    if remote:
        q=c.quota(api)
        (private/'quota-readonly.json').write_text(json.dumps(q,indent=2)+'\n')
        result['quota_snapshot']={k:q[k] for k in ('observed_unix','used_seconds','reserved_seconds','free_seconds','paid_disabled','refresh_time')}
    result['all_six_recoverable_at_observation']=bool(remote and len(result['runs'])==6)
    result['limitation']='Remote inventories and report bytes verified; full checkpoint bytes checked in existing local copies. Future access/retention not guaranteed; no clinical or scientific audit rerun.'
    (out/('recovery-verification.json' if remote else 'recovery-verification-local.json')).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'verified_runs':len(result['runs']),'remote_checked':remote,'new_jobs':0,'quota':result.get('quota_snapshot')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--remote',action='store_true',help='Read-only status/inventory, small report retrieval and quota; never job submission')
    main(parser.parse_args().remote)
