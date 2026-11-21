"""Private hosting fixtures only; no credentials, private downloads or model loads."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from deploy.community_cloud import recovery, app as cloud_app


class Raw:
    def __init__(self,data):self.data=io.BytesIO(data);self.sizes=[]
    def read(self,size,decode_content=False):self.sizes.append(size);return self.data.read(size)


def response(data=b'abc',**kwargs):
    value=Mock(status_code=200,headers={'Content-Length':str(len(data))})
    value.raw=Raw(data)
    for key,val in kwargs.items():setattr(value,key,val)
    context=Mock();context.__enter__=Mock(return_value=value);context.__exit__=Mock(return_value=False)
    return context,value


class PrivateTransport(unittest.TestCase):
    def item(self,data=b'abc'):
        return {'name':'best.pt','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

    def test_exact_scope_sizes_and_original_hashes(self):
        plan=recovery.plan()
        self.assertEqual(len(plan),4);self.assertEqual(sum(x['bytes'] for x in plan),334241994)
        self.assertEqual(plan[0]['version'],'saisathwikregula/trustretina-full-v5-convnext-tiny-c349c706/1')
        self.assertEqual({x['version'] for x in plan[1:]},{'saisathwikregula/trustretina-phase6-analysis-ab40b8f2/1'})
        self.assertEqual(plan[0]['sha256'],'6871e49772543f3463f9d30dd6c758257022463bf562e21342fd154d7aa0eaf1')
        self.assertEqual({x['remote'] for x in plan},{'full/'+x['name'] for x in plan})

    def test_transfer_is_bounded_identity_encoded_and_private(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);context,r=response();get=Mock(return_value=context)
            count=recovery.stream_object(self.item(),'https://www.kaggleusercontent.com/fixture',root/'part',root,get)
            self.assertEqual(count,3);self.assertEqual((root/'part').read_bytes(),b'abc')
            self.assertEqual((root/'part').stat().st_mode & 0o777,0o600)
            self.assertEqual(r.raw.sizes,[3]);self.assertEqual(json.loads((root/'transfer-budget.json').read_text())['charged_bytes'],3)
            self.assertFalse(get.call_args.kwargs['allow_redirects'])
            self.assertEqual(get.call_args.kwargs['headers'],{'Accept-Encoding':'identity'})
            self.assertEqual(get.call_args.kwargs['timeout'],(20,60))

    def test_unapproved_hosts_and_bad_headers_never_consume_a_body(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);get=Mock()
            for url in ('http://www.kaggleusercontent.com/x','https://evil.invalid/x','https://name@www.kaggleusercontent.com/x'):
                with self.assertRaises(ValueError):recovery.stream_object(self.item(),url,root/'part',root,get)
            get.assert_not_called()
            for attrs in ({'status_code':302},{'headers':{}},{'headers':{'Content-Length':'4'}},
                          {'headers':{'Content-Length':'3','Content-Encoding':'gzip'}},
                          {'headers':{'Content-Length':'3','Transfer-Encoding':'chunked'}}):
                context,r=response(**attrs);get.return_value=context
                with self.assertRaises(ValueError):
                    recovery.stream_object(self.item(),'https://www.kaggleusercontent.com/fixture',root/'part',root,get)
                self.assertFalse(r.raw.sizes)
                self.assertFalse((root/'part').exists())

    def test_short_or_wrong_hash_never_accepted_and_budget_survives_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for index,data in enumerate((b'a',b'bad')):
                context,r=response(data,headers={'Content-Length':'3'})
                with self.assertRaises(ValueError):recovery.stream_object(self.item(),'https://www.kaggleusercontent.com/x',root/f'part{index}',root,Mock(return_value=context))
            self.assertEqual(json.loads((root/'transfer-budget.json').read_text())['charged_bytes'],6)
            with patch.object(recovery,'LIMIT',7):
                with self.assertRaises(ValueError):recovery.charge(root,3)
            self.assertEqual(json.loads((root/'transfer-budget.json').read_text())['charged_bytes'],6)

    def test_pinned_metadata_request_and_duplicate_refusal(self):
        reqs=[]
        class Request:pass
        fake_module=SimpleNamespace(ApiListKernelSessionOutputRequest=Request)
        item=recovery.plan()[0]
        api=Mock();client=Mock();api.build_kaggle_client.return_value.__enter__=Mock(return_value=client)
        api.build_kaggle_client.return_value.__exit__=Mock(return_value=False)
        method=client.kernels.kernels_api_client.list_kernel_session_output
        def listing(req):
            reqs.append(req)
            return SimpleNamespace(files=[SimpleNamespace(file_name=item['remote'],url='https://www.kaggleusercontent.com/x')],next_page_token=None)
        method.side_effect=listing
        with patch.dict(sys.modules,{'kagglesdk.kernels.types.kernels_api_service':fake_module}):
            self.assertEqual(recovery.signed_url(api,item),'https://www.kaggleusercontent.com/x')
            self.assertEqual((reqs[0].user_name,reqs[0].kernel_slug,reqs[0].version_label),('saisathwikregula','trustretina-full-v5-convnext-tiny-c349c706','v1'))
            method.side_effect=None;method.return_value=SimpleNamespace(files=[],next_page_token=None)
            with self.assertRaises(ValueError):recovery.signed_url(api,item)

    def test_private_root_refuses_repo_symlink_and_public_modes(self):
        with self.assertRaises(ValueError):recovery.private_directory(recovery.ROOT)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);link=root/'link';link.symlink_to(root,target_is_directory=True)
            with self.assertRaises(ValueError):recovery.private_directory(link)
            public=root/'public';public.mkdir();public.chmod(0o755)
            with self.assertRaises(ValueError):recovery.private_directory(public)

    def test_bundle_is_published_only_after_all_four_hashes_and_contract_verify(self):
        objects=[{'name':n,'bytes':1,'sha256':hashlib.sha256(b'a').hexdigest()} for n in recovery.SIZES]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            def checked(stage):
                self.assertEqual({p.name for p in stage.iterdir()},set(recovery.SIZES)|{'candidate-contract.json'})
            with patch.object(recovery,'plan',return_value=objects),patch.object(recovery,'signed_url',return_value='https://www.kaggleusercontent.com/x'),patch('trustretina.inference.bundle.verify_bundle',side_effect=checked):
                result=recovery.recover(root,Mock(),lambda *a,**kw:response(b'a')[0])
            self.assertEqual(result['transferred_bytes'],4)
            self.assertTrue((root/'inference-v8').is_dir())
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with patch.object(recovery,'plan',return_value=objects),patch.object(recovery,'signed_url',return_value='https://www.kaggleusercontent.com/x'):
                with self.assertRaises(ValueError):recovery.recover(root,Mock(),lambda *a,**kw:response(b'b')[0])
            self.assertFalse((root/'inference-v8').exists())

    def test_sdk_isolation_never_passes_token_in_environment_or_arguments(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);python=root/'recovery-env/bin/python';python.parent.mkdir(parents=True);python.touch()
            run=Mock(side_effect=[SimpleNamespace(returncode=0),SimpleNamespace(returncode=0,stdout='{"status":"PASS"}')])
            with patch.object(cloud_app.subprocess,'run',run),patch.dict(os.environ,{'KAGGLE_API_TOKEN':'fixture-secret'}):
                cloud_app.private_setup(root,'fixture-secret')
            for call in run.call_args_list:
                self.assertNotIn('fixture-secret',str(call.args))
                self.assertNotIn('KAGGLE_API_TOKEN',call.kwargs['env'])
            self.assertEqual(json.loads(run.call_args_list[1].kwargs['input'])['token'],'fixture-secret')

    def test_hosted_binding_does_not_claim_platform_authentication(self):
        from dashboard import readiness
        with patch.object(readiness,'dependency_check',return_value={'status':'PASS'}),patch.object(readiness,'bundle_check',return_value={'status':'PASS'}),patch.object(readiness,'loopback_health',return_value={'status':'PASS'}),patch('dashboard.academic.load_academic_evidence'):
            local=readiness.collect_readiness('fixture',address='0.0.0.0')
            hosted=readiness.collect_readiness('fixture',address='0.0.0.0',platform_backend=True)
            self.assertEqual(local['local_preflight'],'FAIL')
            self.assertEqual(hosted['local_preflight'],'PASS')
            self.assertEqual(hosted['platform_access_control'],'NOT VERIFIED')


@unittest.skipUnless(importlib.util.find_spec('streamlit'),'optional Streamlit')
class CloudSetupPage(unittest.TestCase):
    def test_gate_and_missing_bundle_do_not_download_or_initialize_model(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        path=str(recovery.ROOT/'deploy/community_cloud/app.py')
        with patch.object(st,'secrets',{}),patch.object(cloud_app,'private_setup') as setup:
            page=AppTest.from_file(path).run(timeout=30)
            self.assertFalse(page.exception);self.assertFalse(page.button)
            self.assertTrue(any('not authentication' in x.value for x in page.warning))
            setup.assert_not_called()
        with tempfile.TemporaryDirectory() as td,patch.object(st,'secrets',{'trustretina':{'owner_only_confirmed':True}}),patch.object(recovery,'private_directory',return_value=Path(td)):
            page=AppTest.from_file(path).run(timeout=30)
            self.assertFalse(page.exception)
            next(b for b in page.button if b.label=='Prepare verified original bundle').click().run()
            self.assertTrue(any('API token' in x.value for x in page.error))
            self.assertFalse((Path(td)/'inference-v8').exists())


if __name__=='__main__':unittest.main()
