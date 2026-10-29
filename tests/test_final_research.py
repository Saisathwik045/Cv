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



from trustretina.evaluation.final_research import classification,calibration,weighted_curve,frozen_grid,statistics,bootstrap
from trustretina.reliability.development import component_weights

class FinalMetricTests(unittest.TestCase):
    def test_weighted_two_dimensional_f1_and_accuracy(self):
        y=np.asarray([[1]*8,[0]*8,[0]*8]);p=np.full((3,8),.9);w=np.asarray([.5,.5,1.])
        m=classification(y,p,w)
        self.assertAlmostEqual(m['micro_f1'],.4);self.assertAlmostEqual(m['macro_f1'],.4)
        self.assertEqual(m['labelwise_binary_accuracy'],.25);self.assertEqual(m['subset_exact_match_accuracy'],.25)
        self.assertEqual(m['per_class']['H']['positive'],1);self.assertEqual(m['per_class']['H']['positive_weight'],.5)

    def test_macro_auroc_requires_all_eight_supported(self):
        y=np.zeros((3,8),dtype=int);y[0,:7]=1;m=classification(y,np.full((3,8),.5))
        self.assertIsNone(m['macro_auroc']);self.assertIsNone(m['per_class']['O']['auroc'])
        self.assertEqual(m['per_class']['N']['auroc'],.5)

    def test_ece_and_pair_weighted_loss_known_case(self):
        y=np.asarray([[1]*8,[0]*8]);p=np.full((2,8),.75);m=calibration(y,p,[1.,3.])
        self.assertAlmostEqual(m['macro_ece'],.5)
        self.assertAlmostEqual(m['brier'],(.25**2+3*.75**2)/4)
        self.assertAlmostEqual(m['nll'],-(np.log(.75)+3*np.log(.25))/4)

    def test_weighted_ties_and_saved_cutoff_do_not_retune_on_final_scores(self):
        y=np.asarray([[0]*8,[1]*8,[0]*8]);p=np.full((3,8),.9);score=np.asarray([.9,.9,.1]);w=np.asarray([.5,.5,1.])
        curve=weighted_curve(y,p,score,w);np.testing.assert_array_equal(curve['accepted'],[2,3]);np.testing.assert_array_equal(curve['coverage'],[.5,1.])
        grid=frozen_grid(y,p,score,w,[{'target_coverage':0.,'threshold':None},{'target_coverage':.1,'threshold':.95},{'target_coverage':1.,'threshold':.5}],detailed=True)
        self.assertEqual(grid[1]['accepted'],0);self.assertIsNone(grid[1]['hamming_risk'])
        self.assertEqual(grid[2]['accepted'],2);self.assertEqual(grid[2]['coverage'],.5)

    def test_component_multiplicity_keeps_mass_per_draw(self):
        original=component_weights(['A','A','B']);indices=[0,1,0,1,2];drawn=original[indices]
        self.assertEqual(drawn.sum(),3.);self.assertEqual(drawn[:4].sum(),2.)
        y=np.asarray([[1]*8,[0]*8,[0]*8]);p=np.full((3,8),.9)
        correct=classification(y[indices],p[indices],drawn)
        wrong=classification(y[indices],p[indices],component_weights(['A','A','A','A','B']))
        self.assertNotEqual(correct['hamming_loss'],wrong['hamming_loss'])

if __name__=='__main__':unittest.main()
