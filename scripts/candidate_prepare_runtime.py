"""One GPU-disabled v14 job: copy pinned224, construct only missing native320 roles."""
import json,os,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image
from trustretina.data.validation import file_hash
from trustretina.preprocessing.shared import letterbox_rgb,PreprocessConfig
from trustretina.runner.resumed_review import finish


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(value,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def find_index(root,digest):
    found=[p for p in Path(root).rglob('inputs.json') if file_hash(p)==digest]
    if len(found)!=1:raise ValueError('Pinned index missing/ambiguous')
    return found[0].parent,json.loads(found[0].read_text())


def role_groups(rows,labeled):
    patients={};groups=set()
    for row in rows:
        if ('labels' in row)!=labeled or row['eye'] not in ('left','right'):raise ValueError('Role label/eye contract')
        patients.setdefault(row['patient_id'],[]).append(row);groups.add(row['component_id'])
    for pair in patients.values():
        if len(pair)!=2 or {r['eye'] for r in pair}!={'left','right'} or len({r['component_id'] for r in pair})!=1:raise ValueError('Incomplete/cross-group pair')
        if labeled and pair[0]['labels']!=pair[1]['labels']:raise ValueError('Patient targets differ between eyes')
    return set(patients),groups


def run(spec,*,input_root,output,nonce,source_commit):
    started=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report={'nonce':nonce,'source_commit':source_commit,'training':False,'gpu_requested':False,'stage':'candidate_v14_inputs','status':'incomplete','final_evaluation_access':False,'calibration_labels_in_output':False,'new_training_cache_built':False,'packages':{'numpy':np.__version__,'Pillow':Image.__version__}}
    try:
        if spec['source_dataset']!='andrewmvd/ocular-disease-recognition-odir5k/2' or spec['maximum_seconds']!=3600:raise ValueError('CPU scope changed')
        if report['packages']!={'numpy':'2.1.3','Pillow':'13.0.0'}:raise ValueError('Native preprocessing dependencies changed')
        roots={};indices={}
        for key,item in spec['sources'].items():roots[key],indices[key]=find_index(input_root,item['inputs_sha256'])
        oldtrain=indices['train'];train_path=roots['train']/oldtrain['rows_file']
        if file_hash(train_path)!=oldtrain['rows_sha256']:raise ValueError('Train rows changed')
        train_rows=json.loads(train_path.read_text());seen_p,seen_g=role_groups(train_rows,True)
        result={'schema_version':14,'release_sha256':spec['release_sha256'],'source_dataset':spec['source_dataset'],'source_dataset_version':2,'train_source':spec['sources']['train'],'roles':{},'historical_final_included':False,'calibration_labels_included':False}
        receipts=(output/'native-receipts.jsonl').open('w')
        for role,key in [('validation','validation'),('calibration','calibration')]:
            item=indices[key]['roles'][role];oldroot=roots[key]
            for kind in ('rows','images'):
                if file_hash(oldroot/item[kind+'_file'])!=item[kind+'_sha256']:raise ValueError('Pinned224 role input changed')
            if item['parent_manifest_sha256']!=spec['manifest_sha256'][role]:raise ValueError('Frozen role differs')
            original=json.loads((oldroot/item['rows_file']).read_text())
            if len(original)!={'validation':1050,'calibration':700}[role]:raise ValueError('Role size changed')
            # Only CPU preparation reads calibration labels. Never serialize them into new GPU-mounted output.
            rows=[{k:v for k,v in r.items() if role!='calibration' or k!='labels'} for r in original]
            pp,gg=role_groups(rows,role=='validation')
            if pp&seen_p or gg&seen_g:raise ValueError('Patient/containment leakage between roles')
            seen_p|=pp;seen_g|=gg
            small=output/f'{role}-224.npy';shutil.copyfile(oldroot/item['images_file'],small)
            if file_hash(small)!=item['images_sha256']:raise ValueError('Copied224 bytes changed')
            rowfile=output/f'{role}-rows.json';write(rowfile,rows)
            first=Path(rows[0]['path']);native=[]
            for p in Path(input_root).rglob(first.name):
                candidate=p
                for _ in first.parts:candidate=candidate.parent
                if candidate/first==p and p.is_file():native.append(candidate)
            if len(native)!=1:raise ValueError('Native source root missing/ambiguous')
            native=native[0].resolve();target=output/f'{role}-320.npy'
            array=np.lib.format.open_memmap(target,mode='w+',dtype=np.uint8,shape=(len(rows),3,320,320))
            for i,row in enumerate(rows):
                if time.monotonic()-started>3300:raise TimeoutError('CPU cap: preserved partial outputs; no second job authorized')
                relative=Path(row['path']);p=(native/relative).resolve()
                if relative.parts[0]!='Training Images' or not p.is_relative_to(native) or p.is_symlink() or file_hash(p)!=row['raw_sha256']:raise ValueError('Native image outside frozen allowlist/hash mismatch')
                with Image.open(p) as image:canvas,geometry=letterbox_rgb(image,PreprocessConfig(size=320))
                array[i]=np.asarray(canvas).transpose(2,0,1)
                receipts.write(json.dumps({'role':role,'index':i,'raw_sha256':row['raw_sha256'],'geometry':geometry})+'\n');receipts.flush();os.fsync(receipts.fileno())
                if (i+1)%50==0:
                    array.flush();write(output/'progress.json',{'role':role,'completed':i+1,'expected':len(rows),'GPU_requested':False});print(json.dumps({'role':role,'completed':i+1,'expected':len(rows)}),flush=True)
            array.flush();del array
            result['roles'][role]={'patients':len(pp),'images':len(rows),'components':len(gg),'rows_file':rowfile.name,'rows_sha256':file_hash(rowfile),'original_rows_sha256':item['rows_sha256'],'manifest_sha256':item['parent_manifest_sha256'],'labels_available':role=='validation','resolutions':{str(size):{'file':f'{role}-{size}.npy','sha256':file_hash(output/f'{role}-{size}.npy'),'preprocessing_sha256':PreprocessConfig(size=size).fingerprint()} for size in (224,320)}}
        receipts.close();write(output/'inputs.json',result)
        report.update(status='complete',inputs_sha256=file_hash(output/'inputs.json'),native_images_verified=1750,existing224_reused_exact=True,role_counts={k:{'patients':v['patients'],'images':v['images'],'components':v['components']} for k,v in result['roles'].items()})
    except Exception as e:
        import traceback;traceback.print_exc();report.update(status='failed',error_type=type(e).__name__,error=str(e))
    report['elapsed_seconds']=time.monotonic()-started
    return finish(output,report)
