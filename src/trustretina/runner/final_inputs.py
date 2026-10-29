"""Authorized final-only CPU input preparation. No dataset validation or parameter fitting."""
import json,time,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from trustretina.data.validation import file_hash
from trustretina.preprocessing.shared import letterbox_rgb,PreprocessConfig
from trustretina.reliability.development import canonical_quality
from .resumed_review import finish,progress


def run(spec,*,input_root,output,nonce,source_commit):
    started=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report={'stage':'phase7_inputs','nonce':nonce,'source_commit':source_commit,'training':False,'gpu_requested':False,'status':'incomplete','protocol_sha256':spec['protocol_sha256'],'candidate_contract_sha256':spec['candidate_contract_sha256'],'parameter_fitting':False,'final_evaluation_access':True}
    try:
        if spec['role']!='final_evaluation':raise ValueError('Final-only authorized input role required')
        rows=spec['rows'];first=Path(rows[0]['path']);roots=[]
        for p in Path(input_root).rglob(first.name):
            candidate=p
            for _ in first.parts:candidate=candidate.parent
            if candidate/first==p:roots.append(candidate)
        if len(roots)!=1:raise ValueError('Final original source root missing/ambiguous')
        root=roots[0];array=np.lib.format.open_memmap(output/'final-images.npy',mode='w+',dtype=np.uint8,shape=(len(rows),3,224,224));quality=[]
        for i,row in enumerate(rows):
            if time.monotonic()-started>spec['maximum_seconds']-120:raise TimeoutError('Final input CPU cap')
            p=(root/row['path']).resolve()
            if not p.is_relative_to(root.resolve()) or file_hash(p)!=row['raw_sha256']:raise ValueError('Frozen final image bytes changed')
            with Image.open(p) as image:canvas,_=letterbox_rgb(image)
            array[i]=np.asarray(canvas).transpose(2,0,1);quality.append(canonical_quality(array[i]))
            if (i+1)%100==0:array.flush();progress('final_images_prepared',images=i+1,total=len(rows))
        array.flush();del array
        (output/'final-rows.json').write_text(json.dumps(rows,sort_keys=True));np.savez_compressed(output/'quality-features.npz',features=np.stack(quality))
        checkpoints=[p for p in Path(input_root).rglob('best.pt') if file_hash(p)==spec['checkpoint_sha256']]
        if len(checkpoints)!=1:raise ValueError('Selected verified checkpoint missing/ambiguous')
        shutil.copyfile(checkpoints[0],output/'selected-best.pt')
        index={'role':'final_evaluation','manifest_sha256':spec['manifest_sha256'],'release_sha256':spec['release_sha256'],'protocol_sha256':spec['protocol_sha256'],'candidate_contract_sha256':spec['candidate_contract_sha256'],'preprocessing_sha256':PreprocessConfig().fingerprint(),'images_file':'final-images.npy','images_sha256':file_hash(output/'final-images.npy'),'rows_file':'final-rows.json','rows_sha256':file_hash(output/'final-rows.json'),'quality_file':'quality-features.npz','quality_sha256':file_hash(output/'quality-features.npz'),'checkpoint_file':'selected-best.pt','checkpoint_sha256':file_hash(output/'selected-best.pt'),'patients':len({r['patient_id'] for r in rows}),'images':len(rows)}
        (output/'inputs.json').write_text(json.dumps(index,indent=2,sort_keys=True));report.update(status='complete',inputs_sha256=file_hash(output/'inputs.json'),patients=index['patients'],images=index['images'])
    except Exception as error:
        import traceback;traceback.print_exc();report.update(error_type=type(error).__name__,error=str(error))
    report['elapsed_seconds']=time.monotonic()-started;return finish(output,report)
