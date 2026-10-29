"""Report fixtures test aligned prediction evidence, never remote training claims."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

@unittest.skipUnless(importlib.util.find_spec('matplotlib'),'Optional report environment required')
class FullReportTests(unittest.TestCase):
    def test_actual_npz_comparison_and_misalignment_rejection(self):
        spec=importlib.util.spec_from_file_location('compare_full',Path(__file__).resolve().parents[1]/'scripts/compare_full_experiment.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from trustretina.evaluation.metrics import aggregate_patient_mean
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);base=root/'private';base.mkdir();module.ROOT=root
            names=['a','b'];runs=[];reports={}
            y=np.asarray([[1,0]*4]*2+[[0,1]*4]*2);p=y*.8+.1
            ids=['fixture-a']*2+['fixture-b']*2;eyes=['left','right']*2;components=['fixture-1']*2+['fixture-2']*2
            for name in names:
                folder=root/name;folder.mkdir();runs.append({'status':'complete','backbone':name,'attempts':[{'bundle':name}]})
                np.savez(folder/'validation-images.npz',labels=y,probabilities=p,patient_ids=ids,eyes=eyes,image_ids=['fixture1','fixture2','fixture3','fixture4'],component_ids=components)
                patient=aggregate_patient_mean(ids,eyes,y,p,components)
                np.savez(folder/'validation-patients.npz',labels=patient['labels'],probabilities=patient['probabilities'],patient_ids=patient['patient_ids'])
                (folder/'history.json').write_text(json.dumps([{'epoch':1,'validation_patient_weighted_bce':.1}]))
                reports[folder]={'status':'complete','release_sha256':'fixture-release','config_sha256':'fixture-config','parameter_count':123,'completed_epochs':1,'selection':{'best_epoch':0},'elapsed_seconds':1.,'metrics':{'inference':{'milliseconds_per_image':1.}},'dependencies':{'environment':'synthetic-fixture'}}
            registry={'runs':runs,'config':{'release_manifest_sha256':'fixture-release','backbones':names},'config_sha256':'fixture-config'}
            (base/'registry.json').write_text(json.dumps(registry))
            with patch.object(module,'read_stage',side_effect=lambda folder:(reports[folder],folder,{})):
                module.compare(base,root/'aggregate')
                result=json.loads((root/'aggregate/comparison.json').read_text())
                self.assertEqual(result['selected_backbone'],'a')
                self.assertEqual(result['reports']['b']['patient']['macro_auroc'],1.)
                self.assertTrue((root/'aggregate/validation-auroc.png').is_file())
                self.assertTrue((root/'aggregate/validation-loss.png').is_file())
                # Misordered evidence must fail rather than comparing unrelated rows.
                np.savez(root/'b/validation-images.npz',labels=y,probabilities=p,patient_ids=ids[::-1],eyes=eyes,image_ids=['fixture1','fixture2','fixture3','fixture4'],component_ids=components)
                with self.assertRaisesRegex(ValueError,'alignment'):module.compare(base,root/'aggregate')

if __name__=='__main__':unittest.main()
