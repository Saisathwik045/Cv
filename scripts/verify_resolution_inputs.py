#!/usr/bin/env python3
"""Independent local verification of the recovered TRAIN archive, no image decoding."""
import hashlib,json,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trustretina.runner.verified_artifacts import read_stage
from trustretina.data.validation import file_hash


def verify_archive(archive_path,rows):
    expected={r['path']:r['raw_sha256'] for r in rows}
    if len(expected)!=len(rows):raise ValueError('Duplicate TRAIN raw paths')
    seen=set()
    with tarfile.open(archive_path,'r:') as archive:
        for item in archive:
            if not item.isfile() or item.name not in expected or item.name in seen:
                raise ValueError('Archive contains unexpected/repeated/non-file member')
            stream=archive.extractfile(item);digest=hashlib.sha256()
            while True:
                chunk=stream.read(1024*1024)
                if not chunk:break
                digest.update(chunk)
            if digest.hexdigest()!=expected[item.name]:raise ValueError('Recovered TRAIN image digest mismatch')
            seen.add(item.name)
    if seen!=set(expected):raise ValueError('Incomplete recovered TRAIN archive')
    return len(seen)


def main():
    start=time.monotonic();bundle=ROOT/'artifacts/resolution-v13/preparation'
    report,folder,v=read_stage(bundle)
    if report['status']!='complete' or report['parity_mismatched_images']!=0 or report['native_images_verified']!=4550 or not report['new_320_from_native']:
        raise ValueError('Native/parity prerequisite blocked')
    receipt=json.loads((bundle/'remote-source/kernel-metadata.json').read_text())
    if receipt['enable_gpu'] or not receipt['is_private']:raise ValueError('Preparation must be private and GPU-disabled')
    inputs=json.loads((folder/'inputs.json').read_text());rows=json.loads((folder/inputs['rows_file']).read_text())
    old=json.loads((ROOT/'artifacts/runner/nested-v11/registry.json').read_text())
    if file_hash(folder/inputs['rows_file'])!=old['cache']['train_rows_sha256']:raise ValueError('Rows differ from frozen TRAIN')
    count=verify_archive(folder/inputs['native_archive_file'],rows)
    result={'CPU_job':v['kernel_version'],'source_sha256':v['source_sha256'],'report_sha256':v['artifact_sha256'],
            'source_dataset':'andrewmvd/ocular-disease-recognition-odir5k/2','source_dataset_id':611716,
            'remote_dataset_sources':receipt['dataset_sources'],'raw_images_independently_rehashed_locally':count,
            'patients':len({r['patient_id'] for r in rows}),'parity_images':report['parity_images'],
            'parity_mismatched_images':report['parity_mismatched_images'],'parity_different_channel_values':report['parity_different_channel_values'],
            'parity_tolerance':report['parity_tolerance'],'cache_320_from_originals':True,'preparation_packages':report['packages'],
            'input_index_sha256':file_hash(folder/'inputs.json'),'rows_sha256':inputs['rows_sha256'],
            'resolutions':inputs['resolutions'],'native_archive_sha256':inputs['native_archive_sha256'],
            'initialization':inputs['weights']['convnext_tiny'],'CPU_worker_elapsed_seconds':report['elapsed_seconds'],
            'local_verification_seconds':time.monotonic()-start,'GPU_requested':False,'new_training_submissions':0,
            'originals_recovered_as':'Private TRAIN-only native-train.tar; no unrelated patient archive members allowed',
            'direct_original_download_note':'Provider redirected to a network-blocked storage host; used authorized private Kaggle CPU preparation, with no network-policy bypass.'}
    target=ROOT/'results/resolution-v13/prerequisites.json';target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('CPU_job','raw_images_independently_rehashed_locally','parity_mismatched_images','cache_320_from_originals','local_verification_seconds')}))


if __name__=='__main__':main()
