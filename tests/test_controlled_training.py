"""Focused prospective selection/budget/authorization/resume tests; synthetic only."""
import importlib.util,json,sys,tempfile,tomllib,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
from trustretina.training.controlled import validate_config,advance_auc,require_approval,reserve_attempt,training_oof_thresholds
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import controlled_experiment as controller


def config():return tomllib.loads((ROOT/'configs/controlled-training-v9.toml').read_text())
def plan():return {'config':config(),'plan_sha256':'a'*64,'experiment_source_sha256':'b'*64,'runs':[{**r,'status':'queued','attempts':[]} for r in config()['runs']]}
def approval(p):return {'user_authorized':True,'plan_sha256':p['plan_sha256'],'experiment_source_sha256':p['experiment_source_sha256'],'run_ids':[r['id'] for r in p['runs']],'maximum_gpu_seconds':14400,'approval_reference':'synthetic test only'}


class ControlledTests(unittest.TestCase):
    def test_exact_matrix_and_deferred_factors_fail_closed(self):
        validate_config(config())
        for key,value in [('image_size',320),('final_evaluation_access',True),('calibration_fitting',True),('loss','focal'),('max_epochs',26),('seed',42),('maximum_gpu_total_seconds',20000)]:
            c=config();c[key]=value
            with self.assertRaises(ValueError):validate_config(c)
        c=config();c['runs'].reverse()
        with self.assertRaises(ValueError):validate_config(c)

    def test_auc_checkpoint_and_patience_are_distinct(self):
        state,improved=advance_auc({},.8,0)
        self.assertTrue(improved)
        state,improved=advance_auc(state,.80005,1)
        self.assertTrue(improved);self.assertEqual(state['best_epoch'],1);self.assertEqual(state['bad_epochs'],1)
        state,improved=advance_auc(state,.80005,2)
        self.assertFalse(improved);self.assertEqual(state['best_epoch'],1)
        for epoch in range(3,7):state,_=advance_auc(state,.79,epoch)
        self.assertTrue(state['early_stopped'])
        for bad in (None,float('nan'),float('inf')):
            with self.assertRaises(ValueError):advance_auc({},bad,0)

    def test_no_approval_means_no_external_client_or_submission(self):
        p=plan();api=Mock()
        with self.assertRaises(PermissionError):controller.advance(p,api,submit=True)
        self.assertEqual(api.mock_calls,[])
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);(base/'registry.json').write_text(json.dumps(p))
            with patch.object(controller,'BASE',base),patch.object(sys,'argv',['controlled_experiment.py','advance','--submit']),patch.object(controller.transport,'api_client') as client:
                with self.assertRaises(PermissionError):controller.main()
                client.assert_not_called()

    def test_approval_is_bound_to_plan_source_runs_budget(self):
        p=plan();a=approval(p);require_approval(p,a)
        for key,value in [('plan_sha256','c'*64),('experiment_source_sha256','d'*64),('maximum_gpu_seconds',14401),('run_ids',[]),('user_authorized',False)]:
            bad={**a,key:value}
            with self.assertRaises(PermissionError):require_approval(p,bad)

    def test_six_attempt_global_budget_counts_failed_and_paused(self):
        p=plan()
        for index in (0,0,1,1,2,3):
            run=p['runs'][index];seconds=reserve_attempt(p,run)
            self.assertEqual(seconds,2400)
            run['attempts'].append({'reserved_seconds':seconds,'status':'failed'})
        self.assertEqual(sum(a['reserved_seconds'] for r in p['runs'] for a in r['attempts']),14400)
        with self.assertRaises(ValueError):reserve_attempt(p,p['runs'][2])
        with self.assertRaises(ValueError):reserve_attempt(p,p['runs'][0])

    def test_paid_scaling_or_insufficient_quota_never_submits(self):
        for paid,left in ((False,100),(True,20000)):
            p=plan();api=Mock()
            q={'paid_scaling_disabled':not paid,'free_remaining_seconds':left,'used_seconds':0,'refresh_time':'test'}
            with patch.object(controller,'verify_plan'),patch.object(controller,'mine',return_value=[]),patch.object(controller,'quota',return_value=q):controller.advance(p,api,submit=True,approval=approval(p))
            api.kernels_push.assert_not_called();self.assertIsNotNone(p['queue_blocker'])

    def test_ambiguous_job_is_never_blindly_repeated(self):
        p=plan();p['runs'][0]['status']='ambiguous'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'submission.json').write_text(json.dumps({'submission_attempted':True,'kernel_id':'existing'}))
            p['runs'][0]['attempts']=[{'bundle':str(path),'reserved_seconds':2400}]
            api=Mock()
            with patch.object(controller,'verify_plan'):controller.advance(p,api,submit=True,approval=approval(p))
            api.kernels_push.assert_not_called();self.assertIn('Ambiguous',p['queue_blocker'])

    def test_unknown_remote_round_two_job_blocks_submission(self):
        p=plan();api=Mock()
        with patch.object(controller,'verify_plan'),patch.object(controller,'mine',return_value=['owner/trustretina-round2-v9-existing']):
            with self.assertRaises(ValueError):controller.advance(p,api,submit=True,approval=approval(p))
        api.kernels_push.assert_not_called()

    def test_quota_duration_parser_preserves_raw_and_handles_sdk_fraction(self):
        raw={'gpuQuota':{'timeUsed':'2092.830.0s','timeReserved':'100s','totalTimeAllowed':'21600s','isPayToScaleEnabled':False},'quotaRefreshTime':'test'}
        api=Mock();api.quota_view.return_value=json.dumps(raw)
        with tempfile.TemporaryDirectory() as tmp,patch.object(controller,'BASE',Path(tmp)):
            q=controller.quota(api)
            self.assertAlmostEqual(q['free_remaining_seconds'],21600-2092.83-100)
            self.assertEqual(q['raw'],raw)

    def test_threshold_study_role_guard_and_prospective_tie_rule(self):
        y=np.tile([[0]*8,[1]*8],(12,1));p=np.full(y.shape,.3)
        for role in ('validation','calibration','final_evaluation'):
            with self.assertRaises(PermissionError):training_oof_thresholds(y,p,role=role,positive_components=[12]*8,negative_components=[12]*8,allow_fitting=True)
        thresholds=training_oof_thresholds(y,p,role='train_oof',positive_components=[12]*8,negative_components=[12]*8,allow_fitting=True)
        np.testing.assert_array_equal(thresholds,[.3]*8)
        thresholds=training_oof_thresholds(y,p,role='train_oof',positive_components=[9]*8,negative_components=[12]*8,allow_fitting=True)
        np.testing.assert_array_equal(thresholds,[.5]*8)


@unittest.skipUnless(importlib.util.find_spec('torch'),'optional Torch')
class ControlledResumeTests(unittest.TestCase):
    def test_scheduler_optimizer_rng_epoch_boundary_resume_matches_continuation(self):
        import torch
        from trustretina.training.checkpoint import save_checkpoint,load_checkpoint
        torch.manual_seed(2026);model=torch.nn.Linear(2,8);opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=25,eta_min=1e-6)
        generator=torch.Generator().manual_seed(2026);x=torch.tensor([[.3,.7]]);target=torch.ones(1,8)
        def step(m,o,s):
            o.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(m(x),target);loss.backward();o.step();s.step()
        step(model,opt,scheduler)
        provenance={'backbone':'synthetic_linear','split_sha256':'s','config_sha256':'c','preprocessing_sha256':'p','seed':2026,'data_kind':'synthetic_resume_only'}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last.pt';sha=save_checkpoint(path,model,opt,completed_epoch=0,global_step=1,provenance=provenance,scheduler=scheduler,generator=generator)
            step(model,opt,scheduler);expected={k:v.clone() for k,v in model.state_dict().items()};lr=opt.param_groups[0]['lr']
            other=torch.nn.Linear(2,8);other_opt=torch.optim.AdamW(other.parameters(),lr=1e-4);other_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(other_opt,T_max=25,eta_min=1e-6)
            state=load_checkpoint(path,other,other_opt,expected_sha256=sha,expected_provenance=provenance,scheduler=other_scheduler,generator=generator)
            self.assertEqual(state['next_epoch'],1)
            step(other,other_opt,other_scheduler)
            for k,v in other.state_dict().items():torch.testing.assert_close(v,expected[k],atol=0,rtol=0)
            self.assertEqual(other_opt.param_groups[0]['lr'],lr);self.assertEqual(other_scheduler.last_epoch,scheduler.last_epoch)

    def test_worker_cannot_start_without_execution_approval(self):
        from trustretina.runner.controlled_runtime import run
        cfg=config()
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'full'
            report=run({'config':cfg,'config_sha256':'test','backbone':'convnext_tiny','run_id':'convnext_constant','schedule':'constant','dataset_sources':[],'execution_approved':False},input_root=tmp,output=output,nonce='synthetic-no-submit',source_commit='fixture')
            self.assertEqual(report['status'],'failed');self.assertEqual(report['error_type'],'PermissionError')


class ControllerReceiptTests(unittest.TestCase):
    def test_completed_queue_cannot_resubmit(self):
        p=plan()
        for r in p['runs']:r['status']='complete'
        api=Mock()
        with patch.object(controller,'verify_plan'):controller.advance(p,api,submit=True,approval=approval(p))
        self.assertEqual(api.mock_calls,[])

    def test_active_job_blocks_another_submission(self):
        p=plan();p['runs'][0]['status']='running';api=Mock();api.kernels_status.return_value.status='RUNNING'
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'submission.json').write_text(json.dumps({'kernel_version':'owner/job/1'}))
            p['runs'][0]['attempts']=[{'bundle':str(folder),'reserved_seconds':2400}]
            with patch.object(controller,'verify_plan'):controller.advance(p,api,submit=True,approval=approval(p))
        api.kernels_push.assert_not_called()

    def test_write_ahead_receipt_survives_ambiguous_transport(self):
        p=plan();run=p['runs'][0];run['status']='prepared';api=Mock()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'submission.json').write_text(json.dumps({'submission_attempted':False,'kernel_id':'owner/job'}))
            run['attempts']=[{'bundle':str(folder),'kernel_id':'owner/job','reserved_seconds':2400}]
            def fail(*args,**kwargs):
                self.assertTrue(json.loads((folder/'submission.json').read_text())['submission_attempted'])
                self.assertEqual(json.loads((folder/'registry.json').read_text())['runs'][0]['status'],'submitting')
                raise ConnectionError('synthetic transport interruption')
            api.kernels_push.side_effect=fail
            q={'paid_scaling_disabled':True,'free_remaining_seconds':21600,'used_seconds':0,'refresh_time':'test'}
            with patch.object(controller,'BASE',folder),patch.object(controller,'verify_plan'),patch.object(controller,'mine',return_value=[]),patch.object(controller,'quota',return_value=q):
                with self.assertRaises(ConnectionError):controller.advance(p,api,submit=True,approval=approval(p))
            self.assertEqual(run['status'],'ambiguous')
            self.assertEqual(json.loads((folder/'submission.json').read_text())['submission_error_type'],'ConnectionError')

    def test_prepared_bundle_mounts_only_existing_cache_and_pins_runtime(self):
        p=plan();p.update(config_sha256='config',protocol_sha256='protocol',cache={'inputs_sha256':'inputs','parent_config_sha256':'old-v5','kernel_version':'owner/cache/1'},expected_dependencies={'torch':'pinned'},docker_image='pinned-image',source_commit='fixture')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'artifacts/runner/round2-v9';(root/'src').mkdir();(root/'src/fixture.py').write_text('# synthetic source only\n');(root/'configs').mkdir();(root/'configs/phase1.toml').write_text('# synthetic configuration\n')
            with patch.object(controller,'ROOT',root),patch.object(controller,'BASE',base),patch.object(controller,'verify_plan'):
                bundle,receipt=controller.build(p,p['runs'][0],approval(p))
            metadata=json.loads((bundle/'kernel-metadata.json').read_text())
            self.assertEqual(metadata['dataset_sources'],[]);self.assertEqual(metadata['kernel_sources'],['owner/cache']);self.assertEqual(metadata['docker_image'],'pinned-image')
            self.assertTrue(metadata['is_private']);self.assertFalse(receipt['submission_attempted'])
            self.assertEqual(receipt['execution_timeout_seconds'],1800);self.assertEqual(receipt['budget_reservation_seconds'],2400)
