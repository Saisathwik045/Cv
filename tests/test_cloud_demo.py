"""Synthetic UI/HTTP fixtures; no models, patient records or external networking."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dashboard import guided, readiness
from trustretina.inference.bundle import verified_contract

ROOT = Path(__file__).resolve().parents[1]


def fake_prediction():
    c = verified_contract()
    return {'class_order': guided.CLASSES, 'raw_probabilities': {k: .4 for k in guided.CLASSES},
            'calibrated_probabilities': {k: .3 for k in guided.CLASSES},
            'signals': {'calibrated': {'confidence': .7, 'uncertainty': .001, 'quality': .5}},
            'decision': 'REVIEW REQUIRED', 'device': 'cpu', 'mc_passes': 20, 'threshold': .5,
            'temperature': c['temperature'], 'best_epoch': 2,
            'checkpoint_sha256': c['checkpoint_sha256'], 'preprocessing_sha256': c['preprocessing_sha256']}


class GuidedContracts(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'optional image dependency')
    def test_generated_pair_is_deterministic_decodable_and_replacement_changes_only_left(self):
        from PIL import Image
        pair = guided.synthetic_pair()
        self.assertEqual(pair, guided.synthetic_pair())
        changed = guided.synthetic_pair(True)
        self.assertNotEqual(pair[0], changed[0])
        self.assertEqual(pair[1], changed[1])
        for content in pair:
            with Image.open(io.BytesIO(content)) as image:
                self.assertEqual((image.format, image.size), ('PNG', (320, 240)))

    def test_bad_device_model_or_probabilities_fail_before_explanation(self):
        service = Mock()
        for key, value in [('device', 'cuda'), ('best_epoch', 5), ('checkpoint_sha256', 'wrong')]:
            result = fake_prediction();result[key] = value
            service.predict.return_value = result
            with self.assertRaises(ValueError):
                guided.predict_and_explain(service, (b'left fixture', b'right fixture'))
        result = fake_prediction();result['raw_probabilities']['N'] = float('nan')
        service.predict.return_value = result
        with self.assertRaises(ValueError):
            guided.predict_and_explain(service, (b'left fixture', b'right fixture'))
        service.explain.assert_not_called()

    def test_export_is_explicitly_not_research_evaluation(self):
        output = guided.envelope(fake_prediction())
        self.assertFalse(output['research_evaluation'])
        self.assertEqual(output['input_kind'], 'generated_geometric_non_patient')
        self.assertIn('no disease ground truth', output['interpretation'])
        self.assertNotIn('labels', output)

    def test_guided_reset_does_not_clear_manual_page_state(self):
        state = {'guided_result': {}, 'guided_replacement': True, 'guided_target': 'D', 'prediction': 'manual-page-marker'}
        guided.clear_demo(state)
        self.assertEqual(state, {'prediction': 'manual-page-marker'})


class ReadinessContracts(unittest.TestCase):
    def test_invalid_ports_never_make_requests(self):
        with patch.object(readiness.urllib.request, 'build_opener') as factory:
            for port in ('https://unapproved.invalid', True, 80, 65536):
                self.assertEqual(readiness.loopback_health(port)['status'], 'FAIL')
            factory.assert_not_called()

    def test_health_only_uses_loopback_without_proxies_or_redirects_and_reads_64_bytes(self):
        response = Mock(status=200)
        response.read.return_value = b'ok'
        context = Mock();context.__enter__ = Mock(return_value=response);context.__exit__ = Mock(return_value=False)
        opener = Mock();opener.open.return_value = context
        with patch.object(readiness.urllib.request, 'build_opener', return_value=opener) as factory:
            self.assertEqual(readiness.loopback_health(8501)['status'], 'PASS')
        opener.open.assert_called_once_with('http://127.0.0.1:8501/_stcore/health', timeout=2)
        response.read.assert_called_once_with(64)
        self.assertEqual(factory.call_args.args[0].proxies, {})
        self.assertIsInstance(factory.call_args.args[1], readiness.NoRedirect)
        self.assertIsNone(readiness.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://unapproved.invalid'))

    def test_health_errors_never_expose_raw_exception_text(self):
        opener = Mock();opener.open.side_effect = OSError('secret fixture token')
        with patch.object(readiness.urllib.request, 'build_opener', return_value=opener):
            result = readiness.loopback_health(8501)
        self.assertEqual(result['status'], 'FAIL')
        self.assertNotIn('secret', json.dumps(result))

    def test_dependency_mismatch_is_not_silently_accepted(self):
        pins = {line.split('==')[0]: line.split('==')[1] for line in (ROOT / 'requirements/app-cpu.lock').read_text().splitlines() if line and not line.startswith(('#', '--'))}
        with patch.object(readiness.platform, 'system', return_value='Linux'):
            self.assertEqual(readiness.dependency_check(version=lambda name: pins[name])['status'], 'PASS')
            report = readiness.dependency_check(version=lambda name: 'wrong-version')
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(len(report['mismatches']), len(pins))

    def test_bundle_check_is_hash_only_and_missing_bundle_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(readiness.bundle_check(Path(td))['status'], 'FAIL')
        with patch('trustretina.inference.bundle.verify_bundle', return_value=verified_contract()), patch.dict(sys.modules, {'torch': None}):
            self.assertEqual(readiness.bundle_check('unused-fixture')['verified_files'], 5)

    def test_even_all_local_passes_never_claim_external_or_clinical_readiness(self):
        with patch.object(readiness, 'dependency_check', return_value={'status': 'PASS'}), \
             patch.object(readiness, 'bundle_check', return_value={'status': 'PASS'}), \
             patch.object(readiness, 'loopback_health', return_value={'status': 'PASS'}):
            report = readiness.collect_readiness('fixture')
            bad = readiness.collect_readiness('fixture', address='0.0.0.0', xsrf=False)
        self.assertEqual(report['local_preflight'], 'PASS')
        self.assertEqual(report['live_demonstration_readiness'], 'CONDITIONAL')
        self.assertEqual(report['external_viewing_route'], 'NOT VERIFIED')
        self.assertEqual(report['checkpoint_loading'], 'NOT RUN')
        self.assertFalse(report['clinical_validation'])
        self.assertEqual(bad['local_preflight'], 'FAIL')
        self.assertEqual(bad['live_demonstration_readiness'], 'BLOCKED')
        self.assertNotRegex(json.dumps(report), r'/workspace/|/home/|patient_id|signature=|token=')


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'optional Streamlit')
class CloudDemoPages(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        import numpy as np
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        self.service = Mock()
        self.service.predict.return_value = fake_prediction()
        self.service.explain.return_value = ({'zero_maps': [True, True]}, [np.zeros((224, 224, 3), dtype=np.uint8)] * 2)
        self.constructor = Mock(return_value=self.service)
        module = SimpleNamespace(InferenceService=self.constructor)
        p = patch.dict(sys.modules, {'trustretina.inference.service': module});p.start();self.addCleanup(p.stop)

    def app(self, page):
        from streamlit.testing.v1 import AppTest
        module, render = {'Guided synthetic demo': ('guided', 'render_guided_page'),
                          'Cloud readiness': ('readiness', 'render_readiness_page')}[page]
        app = AppTest.from_string(f'from dashboard.{module} import {render}\n{render}()\n').run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        return app

    def test_explicit_run_export_target_change_replacement_and_reset(self):
        import streamlit as st
        app = self.app('Guided synthetic demo')
        self.constructor.assert_not_called()
        with patch('streamlit.download_button', wraps=st.download_button) as download:
            next(b for b in app.button if b.label == 'Run guided CPU demo').click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(self.constructor.call_args.kwargs, {'device': 'cpu'})
            self.assertEqual(app.dataframe[0].value['Class'].tolist(), guided.CLASSES)
            output = json.loads(download.call_args.args[1]);self.assertFalse(output['research_evaluation'])
            self.assertTrue(any('SYNTHETIC NON-PATIENT' in w.value for w in app.warning))
            app.selectbox[0].set_value('H').run(timeout=30)
            self.assertEqual(self.service.predict.call_count, 1)
            self.assertEqual(self.service.explain.call_count, 1)
            self.assertTrue(any('old target' in i.value for i in app.info))
            next(b for b in app.button if b.label == 'Update synthetic explanation').click().run(timeout=30)
            self.assertEqual(self.service.explain.call_args.args[2], 5)
            next(b for b in app.button if b.label == 'Replace synthetic left image').click().run(timeout=30)
            self.assertNotIn('guided_result', app.session_state.filtered_state)
            self.assertNotIn('guided_cam', app.session_state.filtered_state)
            self.assertTrue(app.session_state['guided_replacement'])
            next(b for b in app.button if b.label == 'Reset synthetic demonstration').click().run(timeout=30)
            self.assertNotIn('guided_replacement', app.session_state.filtered_state)
            self.assertEqual(self.service.predict.call_count, 1)

    def test_failed_run_does_not_publish_old_or_fabricated_result(self):
        app = self.app('Guided synthetic demo')
        app.session_state['guided_result'] = fake_prediction()
        self.service.predict.side_effect = RuntimeError('private fixture path/token')
        next(b for b in app.button if b.label == 'Run guided CPU demo').click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn('guided_result', app.session_state.filtered_state)
        self.assertTrue(any('No substitute' in e.value for e in app.error))
        self.assertFalse(any('path/token' in e.value for e in app.error))

    def test_readiness_page_is_manual_and_never_loads_models(self):
        sample = {'local_preflight': 'PASS', 'checks': {'fixture': {'status': 'PASS'}},
                  'observed_utc': 'synthetic fixture timestamp', 'external_viewing_route': 'NOT VERIFIED'}
        with patch.object(readiness, 'collect_readiness', return_value=sample) as collect:
            app = self.app('Cloud readiness');collect.assert_not_called()
            next(b for b in app.button if b.label == 'Run local readiness checks').click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            collect.assert_called_once()
            self.assertTrue(any('CONDITIONAL' in s.value for s in app.success))
            self.constructor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
