"""Four exact original objects only; isolated child process, no model loading."""
import contextlib
import datetime
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
LIMIT = 668_483_988
SIZES = {'best.pt':334_159_877, 'temperature.json':1_020,
         'quality-normalizer.npz':79_547, 'candidate-pipeline.json':1_550}
HOST = 'www.kaggleusercontent.com'


def plan():
    from trustretina.inference.bundle import verified_contract, required_files
    c=verified_contract();hashes=required_files(c)
    return [{'name':name,'version':c['selected_checkpoint_kernel_version'] if name=='best.pt'
             else c['artifact_storage']['analysis_kernel_version'],
             'remote':'full/'+name,'bytes':size,'sha256':hashes[name]} for name,size in SIZES.items()]


def private_directory(root):
    root=Path(root).absolute()
    if root.is_relative_to(ROOT):
        raise ValueError('Private files must be outside checkout')
    if any(p.is_symlink() for p in (root,*root.parents)):
        raise ValueError('Symlink forbidden')
    root.mkdir(mode=0o700,parents=True,exist_ok=True)
    if root.stat().st_uid!=os.getuid() or root.stat().st_mode & 0o077:
        raise ValueError('Owner-only directory required')
    return root


def charge(root, count):
    """Reserve full response size durably BEFORE requesting any private body."""
    path=root/'transfer-budget.json'
    if path.is_symlink():raise ValueError('Symlink forbidden')
    previous=json.loads(path.read_text())['charged_bytes'] if path.exists() else 0
    if type(previous) is not int or previous<0 or previous+count>LIMIT:
        raise ValueError('Cumulative transfer budget exhausted')
    temp=root/'transfer-budget.next'
    fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump({'charged_bytes':previous+count,'maximum_bytes':LIMIT},f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def stream_object(item, url, target, root, get):
    parsed=urlparse(url)
    if parsed.scheme!='https' or parsed.hostname!=HOST or parsed.username or parsed.password or parsed.port not in (None,443):
        raise ValueError('Unapproved download host')
    charge(root,item['bytes'])
    with get(url,stream=True,timeout=(20,60),allow_redirects=False,
             headers={'Accept-Encoding':'identity'},auth=lambda request:request) as response:
        if (response.status_code!=200 or response.headers.get('Content-Length')!=str(item['bytes'])
                or response.headers.get('Transfer-Encoding')
                or response.headers.get('Content-Encoding','identity').lower() not in ('','identity')):
            raise ValueError('Unexpected response; no private body read')
        digest=hashlib.sha256();count=0
        fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as output:
            while count<item['bytes']:
                block=response.raw.read(min(65536,item['bytes']-count),decode_content=False)
                if not block:raise ValueError('Incomplete private object')
                if len(block)>item['bytes']-count:raise ValueError('Oversized response block')
                count+=len(block);digest.update(block);output.write(block)
            output.flush();os.fsync(output.fileno())
        if digest.hexdigest()!=item['sha256']:raise ValueError('Original SHA256 mismatch')
        return count


def signed_url(api, item):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    owner,slug,version=item['version'].split('/');token='';matches=[]
    with api.build_kaggle_client() as client:
        for _ in range(10):
            request=ApiListKernelSessionOutputRequest()
            request.user_name=owner;request.kernel_slug=slug;request.version_label='v'+version
            request.page_size=100;request.page_token=token
            response=client.kernels.kernels_api_client.list_kernel_session_output(request)
            matches.extend(x for x in response.files or [] if x.file_name==item['remote'])
            token=response.next_page_token
            if not token:break
        else:raise ValueError('Metadata pagination cap')
    if len(matches)!=1:raise ValueError('Pinned object absent or duplicated')
    return matches[0].url


def recover(root, api, get):
    from trustretina.inference.bundle import verify_bundle, CONTRACT_PATH
    root=private_directory(root);destination=root/'inference-v8'
    if destination.exists():verify_bundle(destination);return {'files':4,'transferred_bytes':0,'reused':True}
    if shutil.disk_usage(root).free<2*1024**3:raise ValueError('Require 2 GiB free')
    stage=Path(tempfile.mkdtemp(prefix='.stage-',dir=root));count=0
    for item in plan():
        count+=stream_object(item,signed_url(api,item),stage/item['name'],root,get)
    fd=os.open(stage/'candidate-contract.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as output:output.write(CONTRACT_PATH.read_bytes())
    verify_bundle(stage)
    if destination.exists():raise ValueError('Do not overwrite a bundle')
    stage.rename(destination)
    return {'files':4,'transferred_bytes':count,'reused':False}


def main():
    os.umask(0o077);logging.disable(logging.CRITICAL)
    try:
        request=json.loads(sys.stdin.read())
        if request.get('private_setup_confirmed') is not True:raise ValueError('Explicit setup action required')
        root=private_directory(request['root'])
        # Exclusive process-wide lock covers durable budgeting and atomic publication.
        import fcntl
        fd=os.open(root/'recovery.lock',os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            approval={'scope':'streamlit_private_original_inference_v8','objects':plan(),
                      'authorization_reference':'2026-11-21 user go-ahead plus explicit owner private-setup action',
                      'owner_attestation':True,'platform_authentication_verified_by_helper':False,
                      'download_host':HOST,'maximum_charged_bytes':LIMIT,'minimum_free_bytes':2*1024**3,
                      'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            encoded=json.dumps(approval,sort_keys=True).encode()
            approval_hash=hashlib.sha256(encoded).hexdigest()
            approval_fd=os.open(root/f'approval-{approval_hash}.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(approval_fd,'wb') as f:f.write(encoded)
            os.environ['KAGGLE_API_TOKEN']=request['token']
            config=private_directory(root/'kaggle-runtime')
            os.environ['KAGGLE_CONFIG_DIR']=str(config)
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                import kaggle,requests
                if getattr(kaggle.api,'_authenticated',False) is not True:raise ValueError('Authentication failed')
                result=recover(root,kaggle.api,requests.get)
            result.update(status='PASS',scope='streamlit_private_original_inference_v8',
                          approval_sha256=approval_hash,objects=plan())
            receipt_fd,_=tempfile.mkstemp(prefix='recovery-receipt-',suffix='.json',dir=root)
            with os.fdopen(receipt_fd,'w') as f:json.dump(result,f)
        print(json.dumps(result));return 0
    except BaseException:
        print(json.dumps({'status':'FAIL','reason':'Private recovery stopped; inspect account access, exact source availability, host, size, disk and remaining budget. No fallback or automatic retry.'}));return 1
    finally:
        os.environ.pop('KAGGLE_API_TOKEN',None)


if __name__=='__main__':sys.exit(main())
