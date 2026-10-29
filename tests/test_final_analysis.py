"""Synthetic I/O integration: no fitting and no real final data in fixtures."""
import json,tempfile,tomllib,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
try:
    import matplotlib
    from trustretina.runner.final_analysis import run
    from trustretina.data.validation import file_hash
    AVAILABLE=True
except ImportError:AVAILABLE=False

@unittest.skipUnless(AVAILABLE,'Optional complete CPU/report environment required')
class FinalAnalysisFixtureTests(unittest.TestCase):
    def test_all_endpoints_and_saved_cutoffs_without_any_refit(self):
        cfg=tomllib.loads((Path(__file__).resolve().parents[1]/'configs/final-evaluation-v7.toml').read_text());cfg.update(expected_patients=20,bootstrap_replicates=5,minimum_defined_bootstrap_replicates=4)
        rng=np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);inputs=root/'input/prepared';gpu=root/'input/gpu';old=root/'input/phase6';inputs.mkdir(parents=True);gpu.mkdir();old.mkdir()
            ids=[f'fixture-final-{i:02}' for i in range(20)];components=['group-'+pid for pid in ids];y=np.tile(np.arange(20)%2,(8,1)).T;p=np.where(y,.8,.2).astype(float);p[::5]=1-p[::5]
            rows=[{'patient_id':pid,'eye':eye,'path':pid+'-'+eye+'.png','labels':y[i].tolist(),'component_id':components[i]} for i,pid in enumerate(ids) for eye in ('left','right')]
            (inputs/'rows.json').write_text(json.dumps(rows));np.savez_compressed(inputs/'quality.npz',features=rng.random((40,3)))
            index={'role':'final_evaluation','manifest_sha256':'1'*64,'rows_file':'rows.json','rows_sha256':file_hash(inputs/'rows.json'),'quality_file':'quality.npz','quality_sha256':file_hash(inputs/'quality.npz')};(inputs/'inputs.json').write_text(json.dumps(index))
            np.savez_compressed(gpu/'deterministic.npz',labels=y,probabilities=p,patient_ids=ids,component_ids=components,image_probabilities=np.repeat(p,2,0),image_labels=np.repeat(y,2,0),image_ids=[r['path'] for r in rows],image_patient_ids=[r['patient_id'] for r in rows],image_eyes=[r['eye'] for r in rows])
            np.savez_compressed(gpu/'stochastic.npz',patient_ids=ids,probabilities=np.clip(p[None]+rng.normal(0,.02,(20,20,8)),.001,.999),seeds=np.arange(62026,62046))
            (old/'temperature.json').write_text(json.dumps({'temperature':1.3}));np.savez_compressed(old/'quality-normalizer.npz',sorted_training_feature_distributions=np.sort(rng.random((100,3)),axis=0));(old/'candidate-pipeline.json').write_text('{"fixture":true}')
            names=['confidence','confidence_uncertainty','confidence_uncertainty_quality'];cutoffs=[None,.99,.9,.8,.6,.3,0.]
            development={mode+'__'+name:{'fixed_coverage':[{'target_coverage':target,'threshold':threshold} for target,threshold in zip(cfg['coverage_grid'],cutoffs)]} for mode in ('raw','calibrated') for name in names};(old/'development-comparisons.json').write_text(json.dumps(development))
            contract={'temperature':1.3,'temperature_sha256':file_hash(old/'temperature.json'),'quality_normalizer_sha256':file_hash(old/'quality-normalizer.npz'),'artifact_storage':{'candidate_sha256':file_hash(old/'candidate-pipeline.json')},'mc_validation_seed':62026}
            for folder in (gpu,old):
                report={'status':'complete','model_state_unchanged':True,'candidate_contract_sha256':'2'*64,'parameter_fitting':False,'data_kind':'synthetic_fixture_not_remote_evidence','output_hashes':{f.name:file_hash(f) for f in folder.iterdir()}};(folder/'stage-report.json').write_text(json.dumps(report))
            spec={'config':cfg,'contract':contract,'protocol_sha256':'3'*64,'candidate_contract_sha256':'2'*64,'inputs_sha256':file_hash(inputs/'inputs.json'),'inference_report_sha256':file_hash(gpu/'stage-report.json'),'phase6_analysis_report_sha256':file_hash(old/'stage-report.json'),'final_manifest_sha256':'1'*64,'development_comparisons_sha256':file_hash(old/'development-comparisons.json')}
            with patch('trustretina.reliability.development.fit_shared_temperature',side_effect=AssertionError('Final fitting forbidden')):
                report=run(spec,input_root=root/'input',output=root/'output',nonce='fixture',source_commit='fixture')
            self.assertEqual(report['status'],'complete',report.get('error'))
            self.assertFalse(report['parameter_fitting']);self.assertTrue(report['all_binary_decisions_unchanged'])
            summary=json.loads((root/'output/aggregate-summary.json').read_text());self.assertEqual(summary['deployment_accept_count'],0);self.assertEqual(summary['deployment_review_count'],20)
            self.assertEqual(len(summary['results']['reliability']),6)
            grid=summary['results']['reliability']['raw__confidence']['patient_equal']['saved_validation_grid'];self.assertEqual(grid[1]['accepted'],0)
            self.assertEqual(file_hash(root/'output/temperature.json'),contract['temperature_sha256']);self.assertEqual(file_hash(root/'output/quality-normalizer.npz'),contract['quality_normalizer_sha256'])

if __name__=='__main__':unittest.main()
