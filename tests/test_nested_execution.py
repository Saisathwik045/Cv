import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from drive_nested_queue import seconds,projected_charge,observed_rate_floor_with_guard,next_required


class NestedExecutionAccountingTests(unittest.TestCase):
    def test_quota_duration_and_overhead_are_not_worker_time(self):
        self.assertEqual(seconds('4815.738.0s'),4815.738)
        self.assertEqual(observed_rate_floor_with_guard(1040,1000),1)
        self.assertEqual(observed_rate_floor_with_guard(2040,1000),2)
        self.assertEqual(projected_charge(2),4500)
        self.assertEqual(projected_charge(1),2400)
        with self.assertRaises(ValueError):seconds('unknown')

    def test_charged_counter_or_reserved_budget_can_block_next_session(self):
        p={'quota_baseline':{'used_seconds':1000},'runs':[{'status':'complete','attempts':[{'reserved_seconds':2400}]*5},{'status':'queued','attempts':[]}]}
        self.assertEqual(next_required(p,12000,1),14400)
        self.assertGreater(next_required(p,14000,1),14400)
        self.assertGreater(next_required(p,12000,2),14400)
        p['runs'][1]={'status':'prepared','attempts':[{'reserved_seconds':2400}]}
        self.assertEqual(next_required(p,12000,1),14400)

    def test_checkpoint_audit_requires_inner_history_only(self):
        from summarize_nested_execution import audit_inner_history
        from trustretina.training.controlled import advance_auc
        selection={};history=[]
        for i,score in enumerate((.7,.8,.79995)):
            selection,_=advance_auc(selection,score,i)
            history.append({'epoch':i+1,'inner_patient_macro_auroc':score,'inner_patient_metrics':{'macro_auroc':score}})
        report={'history':history,'selection':selection}
        self.assertEqual(audit_inner_history(report)['best_epoch'],2)
        report['history'][0]['outer_macro_auroc']=.99
        with self.assertRaisesRegex(ValueError,'outer'):audit_inner_history(report)

    def test_provider_name_compaction_preserves_code_and_scientific_identity(self):
        import json,tempfile
        from drive_nested_queue import compact_provider_metadata
        import nested_experiment as c
        with tempfile.TemporaryDirectory() as t:
            bundle=Path(t);code=b'fixture: no executable training';(bundle/'execute.py').write_bytes(code)
            meta={'id':'owner/'+'x'*52,'title':'x'*52,'is_private':True,'enable_gpu':True,'kernel_sources':['fixture-cache']}
            (bundle/'kernel-metadata.json').write_text(json.dumps(meta))
            receipt={'kernel_id':meta['id'],'submission_attempted':False,'nonce':'abc12345'*4,'source_sha256':c.file_hash(bundle/'execute.py')}
            run={'id':'fold2_sqrt_ratio_bce','outer_fold':2,'arm':'sqrt_ratio_bce','attempts':[{'kernel_id':meta['id']}]};plan={}
            compact_provider_metadata(plan,run,bundle,receipt)
            updated=json.loads((bundle/'kernel-metadata.json').read_text())
            self.assertLessEqual(len(updated['title']),50);self.assertEqual(updated['is_private'],True)
            self.assertEqual(updated['kernel_sources'],['fixture-cache']);self.assertEqual((bundle/'execute.py').read_bytes(),code)
            self.assertEqual(run['id'],'fold2_sqrt_ratio_bce');self.assertTrue(receipt['kernel_id'].endswith('abc12345'))
            (bundle/'kernel-metadata.json').write_text(json.dumps(meta));receipt['submission_attempted']=True
            with self.assertRaises(ValueError):compact_provider_metadata(plan,run,bundle,receipt)

    def test_remote_worker_archive_is_hashed_without_execution(self):
        import base64,hashlib,io,json,tempfile,zipfile
        from summarize_nested_execution import verify_embedded_worker
        b=io.BytesIO();source=b'raise RuntimeError("must never execute this verification fixture")'
        with zipfile.ZipFile(b,'w') as z:z.writestr('src/fixture.py',source)
        expected=hashlib.sha256(b'src/fixture.py'+source).hexdigest()
        script='import base64,json\nbase64.b64decode('+repr(base64.b64encode(b.getvalue()).decode())+')\njson.loads('+repr(json.dumps({'run_id':'fixture'}))+')\n'
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'execute.py';p.write_text(script)
            self.assertEqual(verify_embedded_worker(p,expected,{'run_id':'fixture'}),expected)
            with self.assertRaises(ValueError):verify_embedded_worker(p,'wrong',{'run_id':'fixture'})
            with self.assertRaises(ValueError):verify_embedded_worker(p,expected,{'run_id':'another'})
