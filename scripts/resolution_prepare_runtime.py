"""Read-only, TRAIN-allowlisted CPU native-image recovery and dual-resolution cache."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np
from PIL import Image
from trustretina.preprocessing.shared import letterbox_rgb, PreprocessConfig
from trustretina.data.validation import file_hash, canonical_hash


def atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    temporary.replace(path)


def run(spec, *, input_root, output, nonce, source_commit):
    started = time.monotonic(); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    report = {'nonce':nonce, 'source_commit':source_commit, 'training':False, 'gpu_requested':False,
              'stage':'resolution_v13_native_inputs', 'status':'incomplete', 'source_dataset':spec['source_dataset'],
              'source_dataset_version':2, 'old_cache_version':spec['old_cache_version'],
              'calibration_access':False, 'final_evaluation_access':False, 'old_validation_access':False,
              'packages':{'Pillow':Image.__version__, 'numpy':np.__version__}}
    try:
        if spec['source_dataset'] != 'andrewmvd/ocular-disease-recognition-odir5k/2':
            raise ValueError('Only original pinned authorized source permitted')
        matches = [p for p in Path(input_root).rglob('inputs.json') if file_hash(p)==spec['old_inputs_sha256']]
        if len(matches)!=1:raise ValueError('Pinned original cache index missing/ambiguous')
        oldroot = matches[0].parent; old = json.loads(matches[0].read_text()); item = old['roles']['train']
        if old['preprocessing_sha256']!=PreprocessConfig().fingerprint():raise ValueError('Preprocessing identity changed')
        for key in ('images','rows'):
            if file_hash(oldroot/item[key+'_file'])!=item[key+'_sha256']:raise ValueError('Old TRAIN cache hash mismatch')
        rows = json.loads((oldroot/item['rows_file']).read_text())
        if canonical_hash(rows)!=spec['rows_canonical_sha256'] or len(rows)!=4550:
            raise ValueError('Wrong TRAIN allowlist')
        oldimages = np.load(oldroot/item['images_file'], mmap_mode='r', allow_pickle=False)
        if oldimages.shape!=(len(rows),3,224,224) or oldimages.dtype!=np.uint8:raise ValueError('Wrong old TRAIN tensor')
        first = Path(rows[0]['path']); roots=[]
        for path in Path(input_root).rglob(first.name):
            candidate=path
            for _ in first.parts:candidate=candidate.parent
            if candidate/first==path:roots.append(candidate)
        if len(roots)!=1:raise ValueError('Native source root missing/ambiguous')
        root=roots[0]; receipts=[]; mismatches=[]; pixel_differences=0
        target=output/'train-320.npy'
        images=np.lib.format.open_memmap(target, mode='w+', dtype=np.uint8, shape=(len(rows),3,320,320))
        with tarfile.open(output/'native-train.tar','w') as archive:
            for i,row in enumerate(rows):
                if time.monotonic()-started>1650:raise TimeoutError('CPU preparation budget exhausted; preserve progress')
                relative=Path(row['path']); path=(root/relative).resolve()
                if relative.parts[0]!='Training Images' or not path.is_relative_to(root.resolve()) or path.is_symlink():
                    raise ValueError('Native source path outside TRAIN allowlist')
                digest=file_hash(path)
                if digest!=row['raw_sha256']:raise ValueError('Original TRAIN raw hash differs from frozen record')
                with Image.open(path) as image:
                    small,g224=letterbox_rgb(image,PreprocessConfig(size=224))
                    large,g320=letterbox_rgb(image,PreprocessConfig(size=320))
                a=np.asarray(small).transpose(2,0,1); b=np.asarray(large).transpose(2,0,1)
                different=int(np.count_nonzero(a!=oldimages[i])); pixel_differences+=different
                if different:mismatches.append({'index':i, 'path':str(relative), 'different_channel_values':different})
                images[i]=b
                archive.add(path,arcname=str(relative),recursive=False)
                receipts.append({'path':str(relative),'raw_sha256':digest,'image_224_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
                                 'image_320_sha256':hashlib.sha256(b.tobytes()).hexdigest(),'geometry_224':g224,'geometry_320':g320})
                if (i+1)%100==0:
                    images.flush();atomic_json(output/'progress.json',{'completed':i+1,'expected':len(rows),'mismatched_images':len(mismatches), 'training':False, 'gpu_requested':False})
                    print(json.dumps({'stage':'resolution_cpu_inputs','completed':i+1,'total':len(rows)}),flush=True)
        images.flush();del images
        atomic_json(output/'source-receipts.json',receipts);atomic_json(output/'parity-mismatches.json',mismatches)
        report.update(native_images_verified=len(receipts), parity_images=len(rows), parity_mismatched_images=len(mismatches),
                      parity_different_channel_values=pixel_differences, parity_tolerance='exact uint8 equality, zero differing channel values',
                      native_receipts_sha256=file_hash(output/'source-receipts.json'), raw_hashes_against_frozen_rows=True)
        if mismatches:raise ValueError('224 cache parity failed; do not approve resolution study')
        # Byte-identical old tensor may be reused only after every reconstruction matched.
        shutil.copyfile(oldroot/item['images_file'],output/'train-224.npy')
        shutil.copyfile(oldroot/item['rows_file'],output/'train-rows.json')
        w=old['weights']['convnext_tiny']
        if file_hash(oldroot/w['file'])!=w['sha256']:raise ValueError('ImageNet initialization bytes changed')
        shutil.copyfile(oldroot/w['file'],output/w['file'])
        inputs={'schema_version':13,'source_role':'train','patients':len({r['patient_id'] for r in rows}), 'images':len(rows),
                'release_sha256':old['release_sha256'],'source_dataset':spec['source_dataset'], 'old_inputs_sha256':spec['old_inputs_sha256'],
                'fold_manifest_sha256':spec['fold_manifest_sha256'],'rows_file':'train-rows.json','rows_sha256':file_hash(output/'train-rows.json'),
                'resolutions':{str(size):{'file':f'train-{size}.npy','sha256':file_hash(output/f'train-{size}.npy'),
                                        'preprocessing_sha256':PreprocessConfig(size=size).fingerprint()} for size in (224,320)},
                'weights':{'convnext_tiny':w}, 'native_archive_file':'native-train.tar','native_archive_sha256':file_hash(output/'native-train.tar'),
                'source_receipts_sha256':file_hash(output/'source-receipts.json'), 'parity_exact':True}
        atomic_json(output/'inputs.json',inputs);report.update(status='complete',inputs_sha256=file_hash(output/'inputs.json'),new_320_from_native=True)
    except Exception as error:
        report.update(status='blocked',error_type=type(error).__name__,error=str(error))
    report['elapsed_seconds']=time.monotonic()-started
    report['output_hashes']={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file() and p.name not in ('stage-report.json','stage-report.sha256')}
    atomic_json(output/'stage-report.json',report)
    (output/'stage-report.sha256').write_text(file_hash(output/'stage-report.json'))
    return report
