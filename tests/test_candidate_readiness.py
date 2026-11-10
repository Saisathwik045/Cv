"""Synthetic-only v14 leakage, checkpoint-release and financial guards; no GPU jobs."""
import copy,importlib.util,json,subprocess,sys,time,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from candidate_study import audit_roles,train_weights,mean_eyes,calibration_after_freeze,require_approval,next_budget


def rows(prefix,labels=True):
    return [dict(patient_id=prefix+str(i),component_id=prefix+str(i),eye=eye,**({'labels':[i]*8} if labels else {})) for i in (0,1) for eye in ('left','right')]


class CandidateReadinessTests(unittest.TestCase):
    def setUp(self):
        self.roles={'train':rows('t'),'validation':rows('v'),'calibration':rows('c',False)}
        self.p={'plan_sha256':'plan','worker_source_sha256':'source','config_sha256':'config','inputs_sha256':'inputs','runs':[{'id':'v14_candidate_res320_seed2026','image_size':320,'status':'awaiting_approval','attempts':[]},{'id':'v14_reference_res224_seed2026','image_size':224,'status':'awaiting_approval','attempts':[]}]}
        self.q={'paid_disabled':True,'reserved_seconds':0,'free_seconds':14400,'used_seconds':0,'refresh_time':'window','observed_unix':1000}

    def test_roles_exclude_calibration_labels_and_component_crossing(self):
        audit_roles(self.roles)
        bad=copy.deepcopy(self.roles);bad['calibration'][0]['labels']=[0]*8
        with self.assertRaisesRegex(ValueError,'Calibration labels'):audit_roles(bad)
        bad=copy.deepcopy(self.roles)
        for r in bad['validation'][:2]:r['component_id']='t0'
        with self.assertRaisesRegex(ValueError,'crosses roles'):audit_roles(bad)

    def test_weights_count_patients_once_and_only_train(self):
        np.testing.assert_equal(train_weights(self.roles['train']),np.ones(8))
        self.roles['validation']*=5
        np.testing.assert_equal(train_weights(self.roles['train']),np.ones(8))

    def test_mean_eye_probabilities_and_missing_eye_rejection(self):
        p=np.asarray([[.1]*8,[.8]*8,[.2]*8,[.6]*8]);m=mean_eyes(self.roles['train'],p)
        np.testing.assert_allclose(m['probabilities'],np.asarray([[.45]*8,[.4]*8]),rtol=0,atol=1e-15)
        with self.assertRaisesRegex(ValueError,'manufacture'):mean_eyes(self.roles['train'][:-1],p[:-1])

    def test_calibration_loader_is_never_called_before_strict_freeze(self):
        calls=[]
        for report in ({},{'status':'paused'},{'status':'complete','strict_best_reload':True}):
            with self.assertRaises(ValueError):calibration_after_freeze(report,lambda:calls.append(True))
        self.assertEqual(calls,[])
        calibration_after_freeze({'status':'complete','strict_best_reload':True,'selected_checkpoint_sha256':'digest'},lambda:calls.append(True));self.assertEqual(calls,[True])

    def test_approval_checks_every_identity_and_budget_before_api(self):
        with self.assertRaises(PermissionError):require_approval(self.p,None)
        result=subprocess.run([sys.executable,str(ROOT/'scripts/candidate_experiment.py'),'advance','--submit'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('separate exact approval before API access',result.stderr)

    def test_full_free_quota_paid_and_stale_gates(self):
        self.assertEqual(next_budget(self.p,self.q,self.p['runs'][0],now=1000),(7500,7800))
        for patch in ({'paid_disabled':False},{'free_seconds':14399},{'reserved_seconds':1},{'observed_unix':900}):
            with self.assertRaises(ValueError):next_budget(self.p,{**self.q,**patch},self.p['runs'][0],now=1000)

    def test_resumes_consume_slots_and_fixed_cap(self):
        first=self.p['runs'][0];first['status']='paused';first['attempts']=[{'reserved_seconds':7800,'resume_verified':True,'elapsed_seconds':1000,'quota_used_before':0}]
        self.assertEqual(next_budget(self.p,self.q,first,now=1000),(2100,2400))
        first['attempts']+= [{'reserved_seconds':2400,'quota_used_before':0}]
        self.p['runs'][1]['attempts']=[{'reserved_seconds':4200,'quota_used_before':0}]
        with self.assertRaisesRegex(ValueError,'cap reached'):next_budget(self.p,self.q,first,now=1000)

    def test_counter_refresh_and_rate_changes_block_resubmission(self):
        self.p['quota_baseline']={'used_seconds':0,'refresh_time':'old'}
        with self.assertRaisesRegex(ValueError,'window'):next_budget(self.p,self.q,self.p['runs'][0],now=1000)
        self.p.pop('quota_baseline');r=self.p['runs'][0];r['status']='paused';r['attempts']=[{'reserved_seconds':7800,'resume_verified':True,'elapsed_seconds':1000,'quota_used_before':0}]
        with self.assertRaisesRegex(ValueError,'accounting'):next_budget(self.p,{**self.q,'used_seconds':3000},r,now=1000)


if __name__=='__main__':unittest.main()

try:
    import torch
    from torch import nn
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Torch not installed locally; CPU Torch CI exercises these synthetic fixtures')
class CandidateWorkerTorchTests(unittest.TestCase):
    def test_cached_head_predictions_preserve_state_and_eye_alignment(self):
        import tempfile
        from candidate_runtime import features_and_predictions
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__();self.backbone=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten());self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(3,8))
        model=Tiny();before={k:v.clone() for k,v in model.state_dict().items()};r=rows('fixture',False)
        # Scramble the source order: outputs must follow the declared deterministic order.
        r=[r[i] for i in (3,0,2,1)]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic.npy';np.save(path,np.stack([np.full((3,32,32),i*50,dtype=np.uint8) for i in range(4)]))
            result=features_and_predictions(model,path,r,32,torch.device('cpu'),2)
            again=features_and_predictions(model,path,r,32,torch.device('cpu'),2)
        logits,eye,patient,passes,ordered=result
        self.assertEqual([(x['patient_id'],x['eye']) for x in ordered],sorted((x['patient_id'],x['eye']) for x in r))
        np.testing.assert_allclose(patient['probabilities'],np.stack([eye[:2].astype(float).mean(0),eye[2:].astype(float).mean(0)]),rtol=0,atol=0)
        self.assertEqual(passes.shape,(20,2,8));np.testing.assert_array_equal(passes,again[3])
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))
        self.assertTrue(all('labels' not in x for x in ordered))

    def test_strict_checkpoint_reload_restores_rng_scheduler_and_contract(self):
        import tempfile
        from trustretina.training.checkpoint import save_checkpoint,load_checkpoint
        provenance={'backbone':'synthetic_only','split_sha256':'roles','config_sha256':'config','preprocessing_sha256':'pixels','seed':2026,'data_kind':'fixture_only','image_size':320,'worker_source_sha256':'source'}
        model=nn.Linear(2,8);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=25);generator=torch.Generator().manual_seed(2026)
        before={k:v.clone() for k,v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fixture.pt';digest=save_checkpoint(path,model,optimizer,completed_epoch=3,global_step=7,provenance=provenance,scheduler=scheduler,generator=generator)
            expected=torch.rand(4)
            with torch.no_grad():
                for p in model.parameters():p.add_(1)
            state=load_checkpoint(path,model,optimizer,expected_sha256=digest,expected_provenance=provenance,scheduler=scheduler,generator=generator)
            np.testing.assert_array_equal(torch.rand(4).numpy(),expected.numpy());self.assertEqual(state['next_epoch'],4)
            self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))
            with self.assertRaisesRegex(ValueError,'provenance mismatch'):
                load_checkpoint(path,model,optimizer,expected_sha256=digest,expected_provenance={**provenance,'image_size':224},scheduler=scheduler,generator=generator)
