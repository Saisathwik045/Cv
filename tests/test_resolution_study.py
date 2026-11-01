import copy,json,sys,tempfile,tomllib,unittest
from pathlib import Path
from unittest.mock import MagicMock,patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from resolution_study import MATRIX,validate_config,require_approval
import resolution_experiment as c


class ResolutionGuardsTests(unittest.TestCase):
    def setUp(self):self.cfg=tomllib.loads((ROOT/'configs/resolution-v13.toml').read_text())

    def test_fixed_matrix_and_no_approval_never_calls_api(self):
        self.assertEqual(len(MATRIX),6);validate_config(self.cfg)
        changed=copy.deepcopy(self.cfg);changed['loss']='sqrt_ratio_bce'
        with self.assertRaises(ValueError):validate_config(changed)
        api=MagicMock()
        with self.assertRaises(PermissionError):c.advance({},api,submit=True)
        self.assertEqual(api.mock_calls,[])
        with patch.object(sys,'argv',['resolution_experiment.py','advance','--submit']),patch.object(c.transport,'api_client') as client:
            with self.assertRaises(PermissionError):c.main()
            client.assert_not_called()

    def test_approval_binds_exact_source_plan_ids_and_both_caps(self):
        p={'plan_sha256':'p','experiment_source_sha256':'s'}
        a={'user_authorized':True,**p,'run_ids':[r[0] for r in MATRIX],'maximum_gpu_seconds':21600,'maximum_sessions':6,'approval_reference':'fixture'}
        require_approval(p,a)
        for key,value in [('maximum_sessions',7),('maximum_gpu_seconds',21601),('experiment_source_sha256','different')]:
            with self.assertRaises(PermissionError):require_approval(p,{**a,key:value})

    def test_current_insufficient_free_quota_blocks_full_matrix_start(self):
        p={'config':self.cfg,'runs':[{'attempts':[]}]};r={'image_size':224}
        q={'used_seconds':8929.225,'free_seconds':12670.775,'reserved_seconds':0,'paid_disabled':True,'refresh_time':'same'}
        with self.assertRaisesRegex(ValueError,'Full six-hour'):c.next_budget(p,q,r)
        self.assertEqual(c.next_budget(p,{**q,'free_seconds':21600},r),2700)
        with self.assertRaises(ValueError):c.next_budget(p,{**q,'free_seconds':21600,'paid_disabled':False},r)
        with self.assertRaises(ValueError):c.next_budget(p,{**q,'free_seconds':21600,'reserved_seconds':1},r)

    def test_failure_and_changed_charging_count_toward_cap(self):
        a={'reserved_seconds':2700,'quota_used_before':0,'elapsed_seconds':1000,'status':'failed'}
        p={'config':self.cfg,'runs':[{'attempts':[a]*6}],'quota_baseline':{'used_seconds':0,'refresh_time':'same'}}
        q={'used_seconds':4000,'free_seconds':21600,'reserved_seconds':0,'paid_disabled':True,'refresh_time':'same'}
        with self.assertRaisesRegex(ValueError,'Six session'):c.next_budget(p,q,{'image_size':320})
        p['runs']=[{'attempts':[a]}]
        # Counter suggests a higher charge multiplier; do not budget just worker time.
        self.assertEqual(c.next_budget(p,q,{'image_size':320}),17100)
        p['runs']=[{'attempts':[a]*3}]
        with self.assertRaisesRegex(ValueError,'Actual quota'):c.next_budget(p,q,{'image_size':320})

    def test_resolution_report_alias_does_not_imply_a_loss_change(self):
        from report_resolution_results import rename
        self.assertEqual(rename({'ratio_bce':{'candidate_for_future_study':'sqrt_ratio_bce'}}),{'res224':{'candidate_for_future_study':'res320'}})

    def test_native_archive_allowlist_and_digest_reject_unrelated_images(self):
        import hashlib,io,tarfile
        from verify_resolution_inputs import verify_archive
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'native.tar';data=b'synthetic fixture, not a patient image'
            row={'path':'Training Images/fixture_left.jpg','raw_sha256':hashlib.sha256(data).hexdigest()}
            with tarfile.open(p,'w') as z:
                info=tarfile.TarInfo(row['path']);info.size=len(data);z.addfile(info,io.BytesIO(data))
            self.assertEqual(verify_archive(p,[row]),1)
            with self.assertRaises(ValueError):verify_archive(p,[{**row,'raw_sha256':'wrong'}])
            with tarfile.open(p,'a') as z:
                info=tarfile.TarInfo('unrelated.jpg');info.size=len(data);z.addfile(info,io.BytesIO(data))
            with self.assertRaisesRegex(ValueError,'unexpected'):verify_archive(p,[row])

    def test_320_must_use_native_information_not_upscaled_224(self):
        from PIL import Image
        from trustretina.preprocessing.shared import letterbox_rgb,PreprocessConfig
        x=np.indices((601,803)).sum(0)%2*255;x=np.repeat(x[:,:,None],3,axis=2).astype(np.uint8)
        native=Image.fromarray(x);a,_=letterbox_rgb(native,PreprocessConfig(size=224));b,_=letterbox_rgb(native,PreprocessConfig(size=320))
        up=a.resize((320,320),Image.Resampling.BILINEAR)
        self.assertFalse(np.array_equal(np.asarray(up),np.asarray(b)))

    def test_recovery_never_overwrites_newer_or_conflicting_registry(self):
        from restore_resolution_state import exact_write
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'registry.json';exact_write(p,{'state':'running'})
            exact_write(p,{'state':'running'})
            with self.assertRaises(ValueError):exact_write(p,{'state':'queued'})
            self.assertEqual(json.loads(p.read_text()),{'state':'running'})


try:
    import torch
    from torch.utils.data import TensorDataset,DataLoader
except ImportError:torch=None


@unittest.skipIf(torch is None,'Optional CPU training dependencies unavailable')
class ResolutionNumericsTests(unittest.TestCase):
    def test_accumulation_matches_full_batch_including_partial_window(self):
        from resolution_study import train_accumulated_epoch
        torch.manual_seed(2026);x=torch.randn(19,4);y=torch.randint(0,2,(19,8)).float()
        a=torch.nn.Linear(4,8);b=copy.deepcopy(a);criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.arange(1,9).float())
        oa=torch.optim.AdamW(a.parameters(),lr=1e-3);ob=torch.optim.AdamW(b.parameters(),lr=1e-3)
        loss,steps=train_accumulated_epoch(a,DataLoader(TensorDataset(x,y),batch_size=8),oa,criterion,'cpu')
        other,othersteps=train_accumulated_epoch(b,DataLoader(TensorDataset(x,y),batch_size=16),ob,criterion,'cpu',accumulation_steps=1)
        self.assertEqual((steps,othersteps),(2,2));self.assertAlmostEqual(loss,other,places=6)
        for v,w in zip(a.parameters(),b.parameters()):torch.testing.assert_close(v,w,atol=2e-7,rtol=1e-6)

    def test_resume_restores_dropout_loader_optimizer_scheduler_rng_and_resolution(self):
        from resolution_study import train_accumulated_epoch
        from trustretina.training.checkpoint import save_checkpoint,load_checkpoint
        torch.manual_seed(2026);x=torch.randn(19,4);y=torch.randint(0,2,(19,8)).float();data=TensorDataset(x,y)
        model=torch.nn.Sequential(torch.nn.Dropout(.2),torch.nn.Linear(4,8));optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=25,eta_min=1e-6);g=torch.Generator().manual_seed(2026)
        loader=DataLoader(data,batch_size=8,shuffle=True,generator=g);criterion=torch.nn.BCEWithLogitsLoss()
        train_accumulated_epoch(model,loader,optimizer,criterion,'cpu');scheduler.step()
        provenance={'backbone':'fixture','split_sha256':'split','config_sha256':'cfg','preprocessing_sha256':'pp','seed':2026,'data_kind':'synthetic_fixture','image_size':224}
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'last.pt';sha=save_checkpoint(p,model,optimizer,completed_epoch=0,global_step=2,provenance=provenance,generator=g,scheduler=scheduler)
            # save_checkpoint returns its digest; avoid relying on filenames or unverified reloads.
            if not isinstance(sha,str):
                from trustretina.data.validation import file_hash
                sha=file_hash(p)
            train_accumulated_epoch(model,loader,optimizer,criterion,'cpu');scheduler.step();expected=copy.deepcopy(model.state_dict())
            new=copy.deepcopy(model);o=torch.optim.AdamW(new.parameters(),lr=1e-3);s=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=25,eta_min=1e-6);rng=torch.Generator()
            load_checkpoint(p,new,o,expected_sha256=sha,expected_provenance=provenance,generator=rng,scheduler=s)
            train_accumulated_epoch(new,DataLoader(data,batch_size=8,shuffle=True,generator=rng),o,criterion,'cpu');s.step()
            for key in expected:torch.testing.assert_close(expected[key],new.state_dict()[key],rtol=0,atol=0)
            with self.assertRaises(ValueError):load_checkpoint(p,new,o,expected_sha256=sha,expected_provenance={**provenance,'image_size':320},generator=rng,scheduler=s)

    def test_resolution_tensor_shape_and_normalization_match_frozen_formula(self):
        from resolution_runtime import CachedEyes
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'images.npy';x=np.full((2,3,320,320),127,dtype=np.uint8);np.save(p,x)
            rows=[{'labels':[0,1,0,0,0,0,0,0]}]*2;ds=CachedEyes(p,rows,320);value,_=ds[0]
            expected=(127/255-np.asarray([.485,.456,.406]))/np.asarray([.229,.224,.225])
            np.testing.assert_allclose(value[:,0,0],expected,atol=2e-7)
            with self.assertRaises(ValueError):CachedEyes(p,rows,224)


if __name__=='__main__':unittest.main()
