"""Analytical final-evaluation tests, with synthetic fixtures only."""
import unittest,tempfile,json
from pathlib import Path
import numpy as np
try:
    from trustretina.runner.final_inputs import run as prepare
    from trustretina.data.validation import file_hash
    from PIL import Image
    INPUTS=True
except ImportError:INPUTS=False

@unittest.skipUnless(INPUTS,'Optional image/duplicate-runtime dependencies required')
class FinalInputTests(unittest.TestCase):
    def test_final_only_cache_copies_verified_checkpoint_and_preserves_raw_images(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data=root/'sources/dataset/photos';data.mkdir(parents=True)
            rows=[]
            for eye in ('left','right'):
                p=data/('fixture-'+eye+'.png');Image.new('RGB',(80,96),(80,100,120)).save(p)
                rows.append({'patient_id':'fixture-final','eye':eye,'path':'photos/'+p.name,'labels':[1,0]*4,'component_id':'fixture-group','raw_sha256':file_hash(p)})
            checkpoint=root/'sources/model/best.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'synthetic-fixture-not-a-model')
            spec={'role':'final_evaluation','rows':rows,'manifest_sha256':'1'*64,'protocol_sha256':'2'*64,'release_sha256':'3'*64,'candidate_contract_sha256':'4'*64,'checkpoint_sha256':file_hash(checkpoint),'maximum_seconds':1800}
            report=prepare(spec,input_root=root/'sources',output=root/'output',nonce='fixture',source_commit='fixture')
            self.assertEqual(report['status'],'complete',report.get('error'));self.assertEqual(report['images'],2)
            index=json.loads((root/'output/inputs.json').read_text());self.assertEqual(index['checkpoint_sha256'],spec['checkpoint_sha256'])
            self.assertEqual(np.load(root/'output/final-images.npy').shape,(2,3,224,224))
            for row in rows:self.assertEqual(file_hash(data/Path(row['path']).name),row['raw_sha256'])
            spec['role']='validation'
            rejected=prepare(spec,input_root=root/'sources',output=root/'rejected',nonce='fixture',source_commit='fixture')
            self.assertEqual(rejected['status'],'incomplete')

if __name__=='__main__':unittest.main()
