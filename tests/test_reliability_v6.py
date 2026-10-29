"""Analytical Phase 6 contracts; synthetic arrays are never remote execution evidence."""
import unittest
import numpy as np
from trustretina.reliability.development import (canonical_quality,quality_midrank,patient_quality,temperature_probabilities,component_weights,fit_shared_temperature,signals,coverage_curve,fixed_coverage_summary)


class ReliabilityNumericTests(unittest.TestCase):
    def test_quality_excludes_exact_letterbox_padding_and_ties(self):
        content=np.full((3,40,80),80,dtype=np.uint8)
        padded=np.broadcast_to(np.array([124,116,104],dtype=np.uint8)[:,None,None],(3,80,80)).copy();padded[:,20:60]=content
        np.testing.assert_array_equal(canonical_quality(content),canonical_quality(padded))
        np.testing.assert_array_equal(canonical_quality(np.broadcast_to(np.array([124,116,104],dtype=np.uint8)[:,None,None],(3,32,32))),[0,0,0])
        np.testing.assert_array_equal(quality_midrank(np.ones((8,3)),np.ones((2,3))),[.5,.5])

    def test_patient_quality_uses_worse_eye_and_rejects_missing(self):
        rows=[{'patient_id':'fixture','eye':'left'},{'patient_id':'fixture','eye':'right'}]
        np.testing.assert_array_equal(patient_quality(rows,[.8,.2],['fixture']),[.2])
        with self.assertRaises(ValueError):patient_quality(rows[:1],[.2],['fixture'])

    def test_temperature_preserves_patient_aggregation_and_decisions(self):
        # mean(sigmoid eye logits) then calibrate differs from mean calibrated eyes.
        p=np.tile([.1,.8,.2,.3,.4,.5,.6,.7],(2,1));q=temperature_probabilities(p,2.)
        np.testing.assert_array_equal(q>=.5,p>=.5)
        left=np.full((1,8),.99);right=np.full((1,8),.4)
        proper=temperature_probabilities((left+right)/2,2.)
        wrong=(temperature_probabilities(left,2.)+temperature_probabilities(right,2.))/2
        self.assertGreater(abs(proper[0,0]-wrong[0,0]),.01)

    def test_group_weighting_and_single_temperature(self):
        ids=['mixed']*2+['g'+str(i) for i in range(50)]
        w=component_weights(ids);self.assertEqual(w[0],.5);self.assertEqual(w[2],1.)
        # Every label has exactly balanced opposing outcomes, including a mixed group.
        y=np.tile(np.arange(52)%2,(8,1)).T
        p=np.where(y==1,.8,.2).astype(float)
        # Twenty percent deliberately incorrect gives an interior finite optimum.
        p[::5]=1-p[::5]
        fit=fit_shared_temperature(p,y,ids,role='calibration',allow_fitting=True)
        self.assertGreater(fit['temperature'],0);self.assertTrue(fit['optimizer_success'])
        self.assertLessEqual(fit['fitted_component_weighted_nll'],fit['identity_component_weighted_nll'])
        with self.assertRaises(ValueError):fit_shared_temperature(p,y,ids,role='validation',allow_fitting=True)
        with self.assertRaises(ValueError):fit_shared_temperature(p,y,ids,role='calibration')

    def test_variance_is_computed_after_pair_aggregation(self):
        # Anti-correlated eye pass probabilities can have zero patient variance.
        eyes=np.asarray([[[.1]*8,[.9]*8],[[.9]*8,[.1]*8]])
        patients=eyes.mean(1)[:,None,:]
        s=signals(np.full((1,8),.5),patients,[1.])
        self.assertEqual(float(s['uncertainty'][0]),0.)
        self.assertGreater(eyes.var(0).mean(),0.)

    def test_candidate_returns_all_labels_and_never_auto_accepts(self):
        from trustretina.reliability.candidate import candidate_outputs
        p=np.full((2,8),.8);p[0,0]=.9999992;p[0,1]=.9999997
        result=candidate_outputs(p,np.stack([p,p]),[1.,1.],temperature=2.)
        self.assertEqual(len(result['positive_labels'][0]),8)
        self.assertEqual(result['decision'],['REVIEW REQUIRED']*2)
        self.assertEqual(int(result['explanation_target_indices'][0]),1)

    def test_phase6_receipt_restore_refuses_overwrite(self):
        import importlib.util,json,tempfile
        from pathlib import Path
        script=Path(__file__).resolve().parents[1]/'scripts/phase6_experiment.py'
        spec=importlib.util.spec_from_file_location('phase6_restore_fixture',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            module.ROOT=Path(td);module.BASE=module.ROOT/'artifacts/runner/phase6'
            r={'stages':{'inference':{'status':'running','bundle':'artifacts/runner/phase6/inference'}}}
            receipt={'kernel_version':'fixture/private/1','submission_attempted':True}
            module.write(module.ROOT/'results/reliability-v6/run-registry.json',{'registry':r,'receipts':{'inference':receipt}})
            self.assertEqual(module.restore(),r)
            self.assertEqual(json.loads((module.BASE/'inference/submission.json').read_text()),receipt)
            with self.assertRaisesRegex(ValueError,'refusing overwrite'):module.restore()

    def test_ties_are_complete_blocks_zero_coverage_undefined(self):
        y=np.asarray([[0]*8,[1]*8,[0]*8]);p=np.full((3,8),.9);score=np.asarray([.9,.9,.1])
        curve=coverage_curve(y,p,score)
        np.testing.assert_array_equal(curve['accepted'],[2,3]);self.assertAlmostEqual(curve['hamming_aurc'],(2/3)*.5+(1/3)*(2/3))
        summary=fixed_coverage_summary(y,p,score,grid=(0.,.1,1.))
        self.assertIsNone(summary[0]['hamming_risk']);self.assertEqual(summary[1]['accepted'],2)

try:
 import torch
 from torch import nn
 from trustretina.models.backbones import create_backbone
 from trustretina.reliability.stochastic import trained_head_dropout,head_patient_samples
 from trustretina.gradcam.implementation import patient_gradcam
 TORCH=True
except ImportError:TORCH=False


@unittest.skipUnless(TORCH,'Optional Torch environment required')
class ReliabilityTorchTests(unittest.TestCase):
    def test_dropout_context_freezes_normalization_and_restores_modes(self):
        class Fixture(nn.Module):
            def __init__(self):
                super().__init__();self.backbone=nn.Sequential(nn.BatchNorm1d(4));self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(4,8))
        m=Fixture();m.train();before=m.backbone[0].running_mean.clone()
        with trained_head_dropout(m):
            self.assertFalse(m.backbone[0].training);self.assertTrue(m.head[0].training)
            m.backbone(torch.randn(4,4))
        self.assertTrue(m.backbone[0].training);torch.testing.assert_close(before,m.backbone[0].running_mean)
        m.head[0]=nn.Identity()
        with self.assertRaises(ValueError):
            with trained_head_dropout(m):pass

    def test_seeded_head_samples_reproduce_and_restore_rng(self):
        m=nn.Module();m.head=nn.Sequential(nn.Dropout(.2),nn.Linear(4,8));m.eval()
        rows=[{'patient_id':'fixture','eye':eye,'labels':[0]*8,'component_id':'g'} for eye in ('left','right')]
        x=torch.ones(2,4);rng=torch.get_rng_state().clone()
        a=head_patient_samples(m,x,rows,passes=20,base_seed=7)
        torch.testing.assert_close(rng,torch.get_rng_state())
        b=head_patient_samples(m,x,rows,passes=20,base_seed=7)
        np.testing.assert_array_equal(a['probabilities'],b['probabilities']);self.assertGreater(a['probabilities'].var(0).max(),0.)

    def test_gradcam_pair_target_all_positives_hook_cleanup_and_no_parameter_update(self):
        class Fixture(nn.Module):
            def __init__(self):
                super().__init__();self.layer=nn.Conv2d(3,2,1,bias=False);self.fc=nn.Linear(2,8,bias=False)
                with torch.no_grad():self.layer.weight.fill_(.2);self.fc.weight.fill_(.1)
            def forward(self,x):return self.fc(self.layer(x).mean((2,3)))
        m=Fixture();state={k:v.clone() for k,v in m.state_dict().items()}
        result=patient_gradcam(m,torch.ones(2,3,16,16),layer=m.layer)
        self.assertEqual(result['positive_indices'],list(range(8)));self.assertEqual(result['target_index'],0)
        self.assertEqual(result['heatmaps'].shape,(2,16,16));self.assertTrue(np.isfinite(result['heatmaps']).all())
        self.assertEqual(len(m.layer._forward_hooks),0)
        for k,v in m.state_dict().items():torch.testing.assert_close(state[k],v)
        self.assertTrue(all(p.grad is None for p in m.parameters()))

if __name__=='__main__':unittest.main()
