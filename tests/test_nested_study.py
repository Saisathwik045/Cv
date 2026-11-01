import copy,importlib.util,json,math,sys,tempfile,tomllib,unittest
from pathlib import Path
from unittest.mock import MagicMock,patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from trustretina.training.nested import (build_manifest,audit_manifest,loss_weights,outer_after_freeze,
                                        require_approval,reserve_attempt,MATRIX)
from trustretina.data.validation import canonical_hash
from trustretina.evaluation.nested_comparison import paired_summary,bootstrap_replicate,bootstrap_intervals


def fixture():
    cfg=tomllib.loads((ROOT/'configs/nested-loss-v11.toml').read_text());rows=[]
    for i in range(360):
        for eye in ('left','right'):
            rows.append({'patient_id':f'p{i:03}','component_id':'linked' if i<3 else f'g{i:03}',
                         'eye':eye,'labels':[int(i%3==0)]*8,'path':f'fixture/{i}-{eye}'})
    return cfg,rows


class NestedStudyTests(unittest.TestCase):
    def test_nested_partition_separation_outer_coverage_and_immutability(self):
        cfg,rows=fixture();original=copy.deepcopy(rows);m=build_manifest(rows,cfg,'fixture-rows-sha')
        audit_manifest(rows,m,cfg);self.assertEqual(rows,original)
        outer=[]
        for f in m['folds']:
            sets=[{rows[i]['patient_id'] for i in f[r+'_indices']} for r in ('fit','inner','outer')]
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]);outer+=list(sets[2])
            owners=[r for r in ('fit','inner','outer') if any(rows[i]['component_id']=='linked' for i in f[r+'_indices'])]
            self.assertEqual(len(owners),1)
        self.assertEqual(len(outer),360);self.assertEqual(len(set(outer)),360)
        self.assertEqual(m,build_manifest(rows,cfg,'fixture-rows-sha'))
        corrupt=copy.deepcopy(m);corrupt['folds'][0]['outer_indices'].append(corrupt['folds'][0]['fit_indices'][0])
        corrupt['sha256']=canonical_hash({k:v for k,v in corrupt.items() if k!='sha256'})
        with self.assertRaises(ValueError):audit_manifest(rows,corrupt,cfg)

    def test_generalized_allocator_matches_frozen_four_fold_assignments(self):
        from trustretina.data.splits import allocate_components as old
        from trustretina.data.group_folds import allocate_components as new
        patients=[{'patient_id':str(i),'labels':[i%2]*8} for i in range(80)]
        units=[[str(i),str(i+1)] for i in range(0,80,2)]
        c={'seed':2026,'split':{'fractions':[.25]*4,'minimum_positive_patients':1,'minimum_negative_patients':1}}
        self.assertEqual(old(patients,units,c)[0],new(patients,units,c)[0])

    def test_weights_use_fitting_patients_and_only_declared_arm_changes(self):
        _,rows=fixture();fit=[r for r in rows if int(r['patient_id'][1:])<60]
        np.testing.assert_array_equal(loss_weights(fit,'ratio_bce'),[2.]*8)
        np.testing.assert_allclose(loss_weights(fit,'sqrt_ratio_bce'),[math.sqrt(2)]*8)
        with self.assertRaises(ValueError):loss_weights(fit,'focal_search')
        with self.assertRaises(ValueError):loss_weights([r for r in fit if r['labels'][0]==0],'ratio_bce')
        with self.assertRaises(ValueError):build_manifest(rows[:-1],fixture()[0],'fixture')

    def test_outer_pixels_are_deferred_until_complete_strict_reload_and_only_once(self):
        access=MagicMock(return_value='outer-predictions');r={'strict_best_reload':False}
        for complete in (False,True):
            with self.assertRaises(PermissionError):outer_after_freeze(r,complete,access)
        access.assert_not_called();r['strict_best_reload']=True
        with self.assertRaises(PermissionError):outer_after_freeze(r,False,access)
        access.assert_not_called();self.assertEqual(outer_after_freeze(r,True,access),'outer-predictions')
        with self.assertRaises(ValueError):outer_after_freeze(r,True,access)
        access.assert_called_once()

    def test_no_approval_no_api_and_global_attempt_cap_counts_failures(self):
        import nested_experiment as c
        api=MagicMock()
        with self.assertRaises(PermissionError):c.advance({},api,submit=True)
        self.assertEqual(api.mock_calls,[])
        bound={'plan_sha256':'plan','experiment_source_sha256':'source'}
        approved={'user_authorized':True,**bound,'maximum_gpu_seconds':14400,'run_ids':[r[0] for r in MATRIX],'approval_reference':'fixture-user-approval'}
        require_approval(bound,approved)
        with self.assertRaises(PermissionError):require_approval(bound,{**approved,'plan_sha256':'changed'})
        plan={'runs':[{'attempts':[{'status':'failed','reserved_seconds':2400}]*6}]}
        with self.assertRaises(ValueError):reserve_attempt(plan,{'attempts':[]})
        with patch.object(sys,'argv',['nested_experiment.py','advance','--submit']),patch.object(c.transport,'api_client') as client:
            with self.assertRaises(PermissionError):c.main()
            client.assert_not_called()

    def test_running_job_is_not_duplicated_and_polling_is_bounded(self):
        import nested_experiment as c
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);bundle=root/'job';bundle.mkdir();(bundle/'submission.json').write_text(json.dumps({'kernel_version':'private-fixture/1'}))
            p={'runs':[{'id':'fixture','status':'running','attempts':[{'bundle':'job'}]}]};api=MagicMock();api.kernels_status.return_value.status='RUNNING'
            with patch.object(c,'ROOT',root),patch.object(c,'verify_plan',return_value={}):
                c.advance(p,api)
                with self.assertRaisesRegex(ValueError,'interval'):c.advance(p,api)
            api.kernels_status.assert_called_once_with('private-fixture/1');api.kernels_push.assert_not_called()

    def test_paired_summary_alignment_primary_tie_and_addressable_bootstrap(self):
        folds=[]
        for k in range(3):
            y=np.repeat(np.tile([0,1],10)[:,None],8,axis=1);p=np.where(y,.8,.2)
            d={'labels':y,'probabilities':p,'patient_ids':np.array([f'p{k}-{i}' for i in range(20)]),'component_ids':np.array([f'g{k}-{i//2}' for i in range(20)])}
            folds.append({'ratio_bce':d,'sqrt_ratio_bce':copy.deepcopy(d)})
        r=paired_summary(folds);self.assertEqual(r['candidate_for_future_study'],'ratio_bce');self.assertEqual(r['mean_paired_macro_auroc_difference'],0.)
        for index in (0,7):
            a=bootstrap_replicate(folds,index);self.assertEqual(a,bootstrap_replicate(folds,index));self.assertTrue(all(v==0 for v in a.values()))
        corrupt=copy.deepcopy(folds);corrupt[0]['sqrt_ratio_bce']['labels'][0,0]=1
        with self.assertRaisesRegex(ValueError,'mismatch'):paired_summary(corrupt)
        self.assertIsNone(bootstrap_intervals([{'metric':None},{'metric':0}])['metric']['percentile_95_interval'])
        self.assertEqual(bootstrap_intervals([{'metric':0}]*20)['metric']['undefined_replicates'],0)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional Torch CPU fixture checks')
class NestedTorchTests(unittest.TestCase):
    def test_loss_gradient_and_fold_bound_resume_provenance(self):
        import torch
        from trustretina.training.checkpoint import save_checkpoint,load_checkpoint
        cfg,rows=fixture();fit=rows[:120]
        weights=loss_weights(fit,'sqrt_ratio_bce');x=torch.zeros((2,8),requires_grad=True);y=torch.tensor([[1.]*8,[0.]*8])
        value=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights))(x,y);value.backward()
        self.assertTrue(torch.isfinite(value) and torch.isfinite(x.grad).all())
        self.assertAlmostEqual(value.item(),(math.sqrt(2)+1)*math.log(2)/2,places=6)
        model=torch.nn.Linear(2,8);opt=torch.optim.AdamW(model.parameters(),lr=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,25,eta_min=1e-6)
        prov={'backbone':'fixture','split_sha256':'fixture','config_sha256':'fixture','preprocessing_sha256':'fixture','seed':2026,'data_kind':'fixture','outer_fold':0,'arm':'ratio_bce'}
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'last.pt';digest=save_checkpoint(p,model,opt,completed_epoch=0,global_step=1,provenance=prov,scheduler=sched)
            with self.assertRaises(ValueError):load_checkpoint(p,model,opt,expected_sha256=digest,expected_provenance={**prov,'outer_fold':1},scheduler=sched)
            state=load_checkpoint(p,model,opt,expected_sha256=digest,expected_provenance=prov,scheduler=sched);self.assertEqual(state['next_epoch'],1)

    def test_worker_refuses_unapproved_execution_before_data_or_gpu(self):
        from trustretina.runner.nested_runtime import run
        cfg,_=fixture();spec={'config':cfg,'config_sha256':'fixture','release_sha256':'fixture','run_id':MATRIX[0][0],'outer_fold':0,'arm':'ratio_bce','execution_approved':False}
        with tempfile.TemporaryDirectory() as t,patch('torch.cuda.is_available') as gpu:
            r=run(spec,input_root=t,output=Path(t)/'output',nonce='fixture-only',source_commit='fixture-only')
            self.assertEqual(r['status'],'failed');self.assertEqual(r['error_type'],'PermissionError');gpu.assert_not_called()
