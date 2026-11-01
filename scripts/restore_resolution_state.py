#!/usr/bin/env python3
"""Restore pinned private preparation/fold artifacts; no submissions or new splits."""
import contextlib,io,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trustretina.data.validation import file_hash,canonical_hash
import resolution_experiment as c


def exact_write(path,value):
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError('Conflicting saved state; reconcile existing jobs, never overwrite')
    if not path.exists():c.write(path,value)


def main():
    saved=json.loads((ROOT/'results/resolution-v13/queue.json').read_text());plan=saved['registry']
    preparation=ROOT/'artifacts/resolution-v13/preparation'
    receipt=json.loads((ROOT/'results/resolution-v13/preparation-receipt.json').read_text())
    exact_write(preparation/'submission.json',receipt)
    api=c.transport.api_client()
    # Existing, pinned CPU output only. retrieve also verifies source/nonce/all hashes.
    report,folder,_=c.transport.retrieve(api,preparation)
    if report['status']!='complete' or file_hash(folder/'inputs.json')!=plan['cache']['inputs_sha256']:
        raise ValueError('Wrong prepared inputs')
    manifest_path=c.BASE/'fold-manifest.json'
    if not manifest_path.exists():
        inventory=json.loads((ROOT/'results/nested-v11/execution-audit.json').read_text())['artifact_inventory'][0]
        target=ROOT/'artifacts/resolution-v13/frozen-fold-recovery'
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            api.kernels_output(inventory['kernel_version'],path=str(target),file_pattern=r'^full/experiment-spec\.json$',page_size=100)
        spec=target/'full/experiment-spec.json'
        if file_hash(spec)!=inventory['output_sha256']['experiment-spec.json']:raise ValueError('Pinned original fold spec changed')
        manifest=json.loads(spec.read_text())['fold_manifest']
        if manifest['sha256']!=plan['fold_manifest_sha256'] or canonical_hash({k:v for k,v in manifest.items() if k!='sha256'})!=manifest['sha256']:
            raise ValueError('Original frozen fold identity changed')
        exact_write(manifest_path,manifest)
    # Do not erase a newer registry or restore an obsolete approval file.
    if not (c.BASE/'registry.json').exists():exact_write(c.BASE/'registry.json',plan)
    for relative,receipt in saved['receipts'].items():
        target=(ROOT/relative).resolve()
        if not target.is_relative_to(c.BASE.resolve()):raise ValueError('Receipt path outside private experiment')
        exact_write(target/'submission.json',receipt)
    actual=json.loads((c.BASE/'registry.json').read_text());c.verify_plan(actual)
    print(json.dumps({'restored_plan':actual['plan_sha256'],'GPU_submissions':0,'folds_recovered_from_original_pinned_spec':True,'approval_created':False}))


if __name__=='__main__':main()
