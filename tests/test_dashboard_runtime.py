"""Regression checks for observed hosted startup failures; no checkpoint reads."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

from dashboard.runtime import cpu_runtime_check

ROOT = Path(__file__).resolve().parents[1]


class RuntimeChecks(unittest.TestCase):
    def test_unsupported_python_stops_before_cpu_import(self):
        with patch('dashboard.runtime.sys.version_info', (3,14,0)), patch('dashboard.runtime.importlib.import_module') as load:
            result=cpu_runtime_check()
        self.assertEqual(result['code'],'unsupported_python');load.assert_not_called()

    def test_missing_or_broken_dependencies_are_safe_and_not_model_claims(self):
        for exc,code in ((ModuleNotFoundError('secret fixture'), 'missing_cpu_dependency'),
                         (OSError('secret fixture'), 'cpu_dependency_import_failed'),
                         (RuntimeError('secret fixture'), 'cpu_dependency_import_failed')):
            with patch('dashboard.runtime.sys.version_info',(3,12,0)), patch('dashboard.runtime.importlib.import_module',side_effect=exc):
                result=cpu_runtime_check()
            self.assertEqual(result['code'],code)
            self.assertNotIn('secret',str(result))
        with patch('dashboard.runtime.sys.version_info',(3,12,0)),patch('dashboard.runtime.importlib.import_module'):
            result=cpu_runtime_check()
        self.assertEqual(result['status'],'PASS')
        self.assertIn('separate checks',result['detail'])

    def test_missing_bundle_diagnostic_has_no_private_path(self):
        from dashboard.readiness import bundle_check
        with tempfile.TemporaryDirectory() as folder:
            result=bundle_check(folder)
            self.assertEqual(result['code'],'bundle_missing')
            self.assertNotIn(folder,str(result))


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('streamlit','torch','PIL')), 'optional CPU app environment')
class MissingDependencyPages(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        st.cache_resource.clear();self.addCleanup(st.cache_resource.clear)

    def app(self,page):
        from streamlit.testing.v1 import AppTest
        module, render = {'Guided synthetic demo': ('guided', 'render_guided_page'),
                          'Preprocessing inspector': ('inspector', 'render_inspector_page')}[page]
        return AppTest.from_string(f'from dashboard.{module} import {render}\n{render}()\n').run(timeout=30)

    def test_guided_missing_import_clears_old_outputs_without_traceback(self):
        def unavailable(bundle):
            raise ModuleNotFoundError('sensitive fixture')
        with patch('dashboard.guided._load_service',unavailable):
            app=self.app('Guided synthetic demo')
            app.session_state['guided_result']={'stale':True}
            next(b for b in app.button if b.label=='Run guided CPU demo').click().run(timeout=30)
        self.assertFalse(app.exception);self.assertTrue(app.error)
        self.assertNotIn('guided_result',app.session_state.filtered_state)
        self.assertNotIn('sensitive',str([e.value for e in app.error]))

    def test_inspector_missing_import_is_a_visible_error_not_traceback(self):
        with patch('dashboard.inspector.inspect',side_effect=ModuleNotFoundError('sensitive fixture')):
            app=self.app('Preprocessing inspector')
        self.assertFalse(app.exception);self.assertTrue(app.error);self.assertFalse(app.dataframe)

    def test_paired_upload_missing_import_disables_inference(self):
        import io
        from PIL import Image
        from streamlit.testing.v1 import AppTest
        buf=io.BytesIO();Image.new('RGB',(40,30),(80,40,30)).save(buf,format='PNG')
        with patch('streamlit.delta_generator.DeltaGenerator.file_uploader',lambda *a,**k:io.BytesIO(buf.getvalue())), \
             patch('trustretina.inference.service.decode_image',side_effect=ModuleNotFoundError('sensitive fixture')), \
             patch('trustretina.inference.service.InferenceService') as model:
            app=AppTest.from_file(str(ROOT/'streamlit_app.py')).run(timeout=30)
            self.assertFalse(app.exception);self.assertTrue(app.error)
            self.assertTrue(next(b for b in app.button if b.label=='Analyze paired images').disabled)
            model.assert_not_called()

    def test_hosted_runtime_failure_prevents_recovery_and_source_launch(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        report={'status':'FAIL','code':'unsupported_python','detail':'Select Python 3.12'}
        with patch.object(st,'secrets',{'trustretina':{'owner_only_confirmed':True}}), \
             patch('dashboard.runtime.cpu_runtime_check',return_value=report), \
             patch('deploy.community_cloud.recovery.private_directory') as private:
            app=AppTest.from_file(str(ROOT/'deploy/community_cloud/app.py')).run(timeout=30)
            self.assertFalse(app.exception);self.assertTrue(app.error);self.assertFalse(app.button)
            private.assert_not_called()


if __name__=='__main__':unittest.main()
