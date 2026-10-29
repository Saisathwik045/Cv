"""One private CPU preprocessing/ImageNet input cache; train and validation only."""
import hashlib,json,time,urllib.request,re
from pathlib import Path
import numpy as np
from PIL import Image
from trustretina.preprocessing.shared import PreprocessConfig,letterbox_rgb
from trustretina.data.validation import file_hash
from .resumed_review import finish,progress


def prepare_inputs(spec,*,input_root,output,nonce,source_commit):
    from torchvision import models
    started=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report={'nonce':nonce,'source_commit':source_commit,'training':False,'gpu_requested':False,'stage':'full_v5_shared_inputs','status':'incomplete','release_sha256':spec['release_sha256'],'config_sha256':spec['config_sha256'],'final_evaluation_access':False}
    try:
        if set(spec['roles'])!={'train','validation'}:raise ValueError('Full input preparation permits train/validation only')
        first=Path(spec['roles']['train']['rows'][0]['path']);roots=[]
        for p in Path(input_root).rglob(first.name):
            candidate=p
            for _ in first.parts:candidate=candidate.parent
            if candidate/first==p:roots.append(candidate)
        if len(roots)!=1:raise ValueError('Original image root ambiguous')
        root=roots[0];preprocessing=PreprocessConfig();manifest={'release_sha256':spec['release_sha256'],'config_sha256':spec['config_sha256'],'preprocessing_sha256':preprocessing.fingerprint(),'roles':{},'weights':{}}
        for role,data in spec['roles'].items():
            rows=data['rows'];target=output/f'{role}-images.npy';array=np.lib.format.open_memmap(target,mode='w+',dtype=np.uint8,shape=(len(rows),3,224,224))
            for i,r in enumerate(rows):
                p=(root/r['path']).resolve()
                if not p.is_relative_to(root.resolve()) or file_hash(p)!=r['raw_sha256']:raise ValueError('Frozen input byte identity changed')
                with Image.open(p) as im:canvas,_=letterbox_rgb(im,preprocessing);array[i]=np.asarray(canvas).transpose(2,0,1)
                if (i+1)%250==0:array.flush();progress('full_input_cache',role=role,images=i+1,total=len(rows))
            array.flush();del array
            p=output/f'{role}-rows.json';p.write_text(json.dumps(rows,sort_keys=True))
            manifest['roles'][role]={'images_file':target.name,'images_sha256':file_hash(target),'rows_file':p.name,'rows_sha256':file_hash(p),'parent_manifest_sha256':data['sha256'],'images':len(rows),'patients':len({r['patient_id'] for r in rows})}
        for name in spec['backbones']:
            weights=models.get_model_weights(name).IMAGENET1K_V1;url=weights.url;filename=url.rsplit('/',1)[-1];expected=re.search(r'-([0-9a-f]+)\.pth$',filename).group(1)
            p=output/filename;urllib.request.urlretrieve(url,p);digest=file_hash(p)
            if not digest.startswith(expected):raise ValueError('Official ImageNet checkpoint digest prefix mismatch')
            manifest['weights'][name]={'file':filename,'sha256':digest,'enum':str(weights),'url':url};progress('imagenet_weights_ready',backbone=name)
        p=output/'inputs.json';p.write_text(json.dumps(manifest,sort_keys=True,indent=2));report.update(status='complete',inputs_sha256=file_hash(p),training_patients=manifest['roles']['train']['patients'],validation_patients=manifest['roles']['validation']['patients'],preprocessing_sha256=preprocessing.fingerprint())
    except Exception as e:report.update(error_type=type(e).__name__,error=str(e))
    report['elapsed_seconds']=round(time.monotonic()-started);return finish(output,report)
