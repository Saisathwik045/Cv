"""Small analytical tests of experimental selection, cache equivalence, and queue safety."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from trustretina.training.comparison import training_weights, patient_loss, advance_selection, comparison_metrics, select_backbone


class FullComparisonTests(unittest.TestCase):
    def test_patient_weights_count_patients_and_reject_conflicts(self):
        a={'patient_id':'a','labels':[1]*8};b={'patient_id':'b','labels':[0]*8};c={'patient_id':'c','labels':[0]*8}
        np.testing.assert_array_equal(training_weights([a,a,b,b,c,c]),[2]*8)
        with self.assertRaises(ValueError):training_weights([a,{**a,'labels':[0]*8},b])
        with self.assertRaises(ValueError):training_weights([a,a])

    def test_patient_loss_matches_analytic_weighted_binary_loss(self):
        self.assertAlmostEqual(patient_loss([[1]*8,[0]*8],[[.5]*8,[.5]*8],[3]*8),2*np.log(2))

    def test_best_loss_and_patience_are_separate_and_ties_keep_earliest(self):
        state={}
        for epoch,loss in enumerate([1.,.99995,.99994,.99993]):
            state,best=advance_selection(state,loss,epoch,min_delta=.0001,patience=3)
            self.assertTrue(best)
        self.assertEqual(state['best_epoch'],3)
        self.assertTrue(state['early_stopped'])
        state,best=advance_selection(state,.99993,4,min_delta=.0001,patience=3)
        self.assertFalse(best);self.assertEqual(state['best_epoch'],3)
        with self.assertRaises(ValueError):advance_selection({},float('nan'),0,min_delta=.0001,patience=3)

    def test_correct_multilabel_accuracy_f1_and_undefined_ranking(self):
        y=np.asarray([[1,0]*4,[0,1]*4]);p=np.full((2,8),.9)
        m=comparison_metrics(y,p)
        self.assertAlmostEqual(m['macro_f1'],2/3);self.assertAlmostEqual(m['micro_f1'],2/3)
        self.assertEqual(m['subset_exact_match_accuracy'],0.)
        self.assertEqual(m['labelwise_binary_accuracy'],.5)
        self.assertEqual(m['macro_auroc'],.5)
        missing=comparison_metrics(np.zeros((2,8)),np.zeros((2,8)))
        self.assertIsNone(missing['macro_auroc']);self.assertIsNone(missing['per_class']['H']['auroc'])

    def test_backbone_selection_requires_all_runs_and_keeps_fixed_tie_order(self):
        order=['b','a'];r=lambda s:{'patient':{'macro_auroc':s}}
        self.assertIsNone(select_backbone(order,{'b':r(.8)}))
        self.assertEqual(select_backbone(order,{'b':r(.8),'a':r(.8)}),'b')
        self.assertIsNone(select_backbone(order,{'b':r(None),'a':r(.8)}))

    def test_queue_does_not_submit_ambiguous_attempt(self):
        spec=importlib.util.spec_from_file_location('full_experiment',Path(__file__).resolve().parents[1]/'scripts/full_experiment.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        class API:
            def kernels_status(self,identity):raise RuntimeError('remote unavailable')
            def kernels_push(self,*a,**k):raise AssertionError('must never submit')
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);module.write(folder/'submission.json',{'kernel_id':'private/id','submission_attempted':True})
            r={'runs':[{'status':'ambiguous','attempts':[{'bundle':str(folder)}]}]}
            with self.assertRaises(RuntimeError):module.advance(folder,r,API(),True)

    def test_restore_recovers_receipts_without_overwriting_existing_state(self):
        spec=importlib.util.spec_from_file_location('full_experiment_restore',Path(__file__).resolve().parents[1]/'scripts/full_experiment.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            module.ROOT=Path(td)
            base=module.ROOT/'artifacts/runner/full-v5'
            relative='artifacts/runner/fixture-attempt'
            receipt={'kernel_version':'fixture/private/1','submission_attempted':True}
            registry={'runs':[{'status':'running','attempts':[{'bundle':relative}]}]}
            module.write(module.ROOT/'results/full-v5/run-registry.json',{'registry':registry,'receipts':{relative:receipt}})
            restored=module.restore(base)
            self.assertEqual(restored,registry)
            self.assertEqual(json.loads((module.ROOT/relative/'submission.json').read_text()),receipt)
            with self.assertRaisesRegex(ValueError,'refusing overwrite'):module.restore(base)

    def test_cache_normalization_matches_shared_preprocessing(self):
        try:
            import torch
            from PIL import Image
            from trustretina.runner.full_runtime import CachedEyes
            from trustretina.preprocessing.shared import preprocess,letterbox_rgb
        except ImportError:self.skipTest('Torch runtime not installed')
        image=Image.fromarray(np.arange(117*83*3,dtype=np.uint8).reshape(117,83,3))
        expected,_=preprocess(image)
        canvas,_=letterbox_rgb(image)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'images.npy';np.save(path,np.asarray(canvas).transpose(2,0,1)[None])
            actual,_=CachedEyes(path,[{'labels':[0]*8}])[0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
