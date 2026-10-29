"""Synthetic end-to-end CPU analysis; fixtures are never Kaggle evidence."""
import json,tempfile,tomllib,unittest
from pathlib import Path
import numpy as np
try:
    import matplotlib
    from trustretina.runner.reliability_analysis import run
    from trustretina.data.validation import file_hash
    AVAILABLE=True
except ImportError:AVAILABLE=False

@unittest.skipUnless(AVAILABLE,'Optional complete CPU/report environment required')
class AnalysisFixtureTests(unittest.TestCase):
    def test_aligned_private_arrays_fit_once_and_keep_review_default(self):
        root_config=Path(__file__).resolve().parents[1]/'configs/reliability-v6.toml'
        cfg=tomllib.loads(root_config.read_text());cfg['checkpoint_sha256']='0'*64;cfg['release_sha256']='1'*64
        rng=np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);inputs=root/'input/prepared';gpu=root/'input/gpu';inputs.mkdir(parents=True);gpu.mkdir(parents=True)
            index={'protocol_sha256':'2'*64,'roles':{}}
            feature={'train':rng.random((100,3))}
            for role,prefix in (('validation','v'),('calibration','c')):
                ids=[f'fixture-{prefix}{i:02}' for i in range(60)];components=['group-'+pid for pid in ids]
                y=np.tile(np.arange(60)%2,(8,1)).T;p=np.where(y==1,.8,.2).astype(float);p[::5]=1-p[::5]
                rows=[]
                for i,pid in enumerate(ids):
                    for eye in ('left','right'):rows.append({'patient_id':pid,'eye':eye,'path':pid+'-'+eye+'.png','labels':y[i].tolist(),'component_id':components[i]})
                rowpath=inputs/(role+'-rows.json');rowpath.write_text(json.dumps(rows));index['roles'][role]={'rows_file':rowpath.name,'rows_sha256':file_hash(rowpath),'parent_manifest_sha256':'3'*64}
                feature[role]=rng.random((120,3));mc=np.clip(p[None]+rng.normal(0,.03,(20,60,8)),.001,.999)
                np.savez_compressed(gpu/(role+'.npz'),patient_ids=ids,component_ids=components,labels=y,probabilities=p,image_probabilities=np.repeat(p,2,axis=0),image_ids=[r['path'] for r in rows],image_patient_ids=[r['patient_id'] for r in rows],image_eyes=[r['eye'] for r in rows],mc_patient_probabilities=mc,mc_seeds=np.arange(cfg['mc_'+role+'_seed'],cfg['mc_'+role+'_seed']+20))
            np.savez_compressed(inputs/'quality-features.npz',**feature);index.update(quality_file='quality-features.npz',quality_sha256=file_hash(inputs/'quality-features.npz'))
            (inputs/'inputs.json').write_text(json.dumps(index));(gpu/'gradcam-records.json').write_text('[]')
            inference={'status':'complete','protocol_sha256':'2'*64,'checkpoint_sha256':'0'*64,'model_state_unchanged':True,'gradcam_zero_maps':0,'data_kind':'synthetic_fixture_not_remote_evidence','output_hashes':{p.name:file_hash(p) for p in gpu.iterdir()}}
            (gpu/'stage-report.json').write_text(json.dumps(inference))
            spec={'config':cfg,'config_sha256':'4'*64,'protocol_sha256':'2'*64,'release_sha256':'1'*64,'checkpoint_sha256':'0'*64,'inputs_sha256':file_hash(inputs/'inputs.json'),'inference_report_sha256':file_hash(gpu/'stage-report.json')}
            report=run(spec,input_root=root/'input',output=root/'output',nonce='fixture',source_commit='fixture')
            self.assertEqual(report['status'],'complete',report.get('error'))
            candidate=json.loads((root/'output/candidate-pipeline.json').read_text())
            self.assertEqual(candidate['deployment_default'],'REVIEW REQUIRED');self.assertFalse(candidate['automatic_acceptance_policy_fitted'])
            self.assertFalse(report['final_evaluation_access']);self.assertTrue(report['all_deterministic_binary_decisions_unchanged'])
            self.assertGreater(report['temperature'],0)
            summary=json.loads((root/'output/aggregate-summary.json').read_text())
            self.assertEqual(len(summary['development_comparisons']),6)
            self.assertNotIn('patient_ids',json.dumps(summary))

if __name__=='__main__':unittest.main()
