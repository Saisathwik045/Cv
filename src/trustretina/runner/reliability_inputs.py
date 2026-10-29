"""CPU preparation of allowed calibration pixels and training-only quality normalizers."""
import json,time,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from trustretina.data.validation import file_hash
from trustretina.preprocessing.shared import letterbox_rgb,PreprocessConfig
from trustretina.reliability.development import canonical_quality
from .resumed_review import finish,progress


def prepare(spec,*,input_root,output,nonce,source_commit):
    started=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report={'nonce':nonce,'source_commit':source_commit,'training':False,'gpu_requested':False,'stage':'phase6_inputs','status':'incomplete','protocol_sha256':spec['protocol_sha256'],'protocol_commit_before_calibration_access':spec['protocol_commit'],'release_sha256':spec['release_sha256'],'final_evaluation_access':False}
    try:
        if set(spec['roles'])!={'validation','calibration'}:raise ValueError('Phase6 allows validation/calibration only')
        matches=[p for p in Path(input_root).rglob('inputs.json') if file_hash(p)==spec['previous_inputs_sha256']]
        if len(matches)!=1:raise ValueError('Verified Phase5 shared cache missing/ambiguous')
        old=matches[0].parent;cache=json.loads(matches[0].read_text());index={'roles':{},'protocol_sha256':spec['protocol_sha256'],'release_sha256':spec['release_sha256'],'preprocessing_sha256':PreprocessConfig().fingerprint()}
        features={}
        for role in ('train','validation'):
            item=cache['roles'][role];path=old/item['images_file']
            if file_hash(path)!=item['images_sha256'] or file_hash(old/item['rows_file'])!=item['rows_sha256']:raise ValueError('Phase5 cache changed')
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            features[role]=np.stack([canonical_quality(x) for x in array]);progress('quality_features',role=role,images=len(array))
            if role=='validation':
                shutil.copyfile(path,output/'validation-images.npy')
                oldrows=json.loads((old/item['rows_file']).read_text())
                if oldrows!=spec['roles']['validation']['rows']:raise ValueError('Frozen validation alignment changed')
        rows=spec['roles']['calibration']['rows'];first=Path(rows[0]['path']);roots=[]
        for p in Path(input_root).rglob(first.name):
            candidate=p
            for _ in first.parts:candidate=candidate.parent
            if candidate/first==p:roots.append(candidate)
        if len(roots)!=1:raise ValueError('Original source root missing/ambiguous')
        root=roots[0];path=output/'calibration-images.npy';array=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=(len(rows),3,224,224));q=[]
        for i,row in enumerate(rows):
            if time.monotonic()-started>spec['maximum_seconds']-90:raise TimeoutError('CPU input cap')
            p=(root/row['path']).resolve()
            if not p.is_relative_to(root.resolve()) or file_hash(p)!=row['raw_sha256']:raise ValueError('Frozen calibration image identity changed')
            with Image.open(p) as image:canvas,_=letterbox_rgb(image)
            array[i]=np.asarray(canvas).transpose(2,0,1);q.append(canonical_quality(array[i]))
            if (i+1)%100==0:array.flush();progress('calibration_cache',images=i+1,total=len(rows))
        array.flush();del array;features['calibration']=np.stack(q)
        for role,item in spec['roles'].items():
            p=output/f'{role}-rows.json';p.write_text(json.dumps(item['rows'],sort_keys=True))
            images=output/f'{role}-images.npy'
            index['roles'][role]={'images_file':images.name,'images_sha256':file_hash(images),'rows_file':p.name,'rows_sha256':file_hash(p),'parent_manifest_sha256':item['manifest_sha256'],'images':len(item['rows']),'patients':len({r['patient_id'] for r in item['rows']})}
        np.savez_compressed(output/'quality-features.npz',**features)
        index['quality_file']='quality-features.npz';index['quality_sha256']=file_hash(output/'quality-features.npz');index['quality_normalizer_source']='all frozen training cache images, no labels'
        (output/'inputs.json').write_text(json.dumps(index,indent=2,sort_keys=True));report.update(status='complete',inputs_sha256=file_hash(output/'inputs.json'),roles={k:{'patients':v['patients'],'images':v['images']} for k,v in index['roles'].items()},packages={'numpy':np.__version__,'Pillow':Image.__version__})
    except Exception as e:
        import traceback;traceback.print_exc();report.update(error_type=type(e).__name__,error=str(e))
    report['elapsed_seconds']=time.monotonic()-started;return finish(output,report)
