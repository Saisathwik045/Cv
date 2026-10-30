import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from trustretina.app.state import clear_patient, reconcile_uploads
from trustretina.inference.bundle import verified_contract, verify_bundle
from trustretina.app.results import load_evidence


class DemoContracts(unittest.TestCase):
    def test_reset_clears_patient_and_widget_state(self):
        state = {'left_0': b'private', 'right_0': b'private', 'prediction': {'p': 1}, 'explanation': [1], 'pair_signature': ('x', 'y')}
        clear_patient(state)
        self.assertEqual(state, {'upload_generation': 1})
        clear_patient(state)
        self.assertEqual(state, {'upload_generation': 2})

    def test_changed_or_missing_upload_invalidates_results(self):
        state = {}
        reconcile_uploads(state, b'left', b'right')
        state['prediction'] = 'previous'
        reconcile_uploads(state, b'left', b'right')
        self.assertIn('prediction', state)
        reconcile_uploads(state, b'left', None)
        self.assertNotIn('prediction', state)

    def test_contract_and_real_aggregate_sources(self):
        contract = verified_contract()
        self.assertEqual(contract['class_order'], ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
        table, final, report = load_evidence()
        self.assertEqual(len(table), 5)
        self.assertEqual(final['patients'], 350)
        self.assertIn('REVIEW REQUIRED', report)
        self.assertFalse(final['policy_fitting'])

    def test_no_missing_bundle_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, 'random weights are never used'):
                verify_bundle(folder)

    def test_tampered_contract_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'candidate-contract.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verified_contract(path)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('torchvision'), 'optional Torch environment')
class DemoImages(unittest.TestCase):
    def test_invalid_empty_oversize_and_valid_image(self):
        from PIL import Image
        from trustretina.inference.service import decode_image
        for content in (None, b'', b'not image', b'x' * (20 * 1024 * 1024 + 1)):
            with self.assertRaises(ValueError):
                decode_image(content)
        b = io.BytesIO()
        Image.new('RGB', (32, 24), 'gray').save(b, format='PNG')
        self.assertEqual(decode_image(b.getvalue()).size, (32, 24))

    def test_missing_eye_rejected_before_model_use(self):
        from trustretina.inference.service import InferenceService
        service = object.__new__(InferenceService)
        with self.assertRaisesRegex(ValueError, 'Both left and right'):
            service.prepare(None, b'x')

    def test_gradcam_selectable_target_and_state(self):
        import torch
        from torch import nn
        from trustretina.gradcam.implementation import patient_gradcam
        torch.manual_seed(5)
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 1)
                self.head = nn.Linear(4, 8)
            def forward(self, x):
                return self.head(self.conv(x).mean((2, 3)))
        model = Tiny().eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = patient_gradcam(model, torch.ones(2, 3, 8, 8), layer=model.conv, target_index=3)
        self.assertEqual(result['target_index'], 3)
        self.assertEqual(result['heatmaps'].shape, (2, 8, 8))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        for target in (-1, 8, 'D'):
            with self.assertRaises(ValueError):
                patient_gradcam(model, torch.ones(2, 3, 8, 8), layer=model.conv, target_index=target)


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'optional Streamlit environment')
class DemoPage(unittest.TestCase):
    def test_startup_results_and_reset(self):
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(str(root / 'streamlit_app.py')).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.button[0].label == 'Analyze paired images' or any(x.label == 'Analyze paired images' for x in app.button))
        self.assertTrue(next(x for x in app.button if x.label == 'Analyze paired images').disabled)
        app.sidebar.radio[0].set_value('Research results').run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.dataframe), 1)
        app.session_state['prediction'] = {'private': 'session'}
        next(x for x in app.button if x.label == 'Reset patient session').click().run(timeout=30)
        self.assertNotIn('prediction', app.session_state.filtered_state)


class ResetStorageTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('streamlit'), 'optional Streamlit')
    def test_reset_only_purges_current_session_storage(self):
        from unittest.mock import patch, Mock
        from trustretina.app.state import reset_streamlit_session
        context = Mock(session_id='current-session')
        manager = Mock()
        instance = Mock(media_file_mgr=manager)
        state = {'prediction': {'sensitive': True}}
        with patch('streamlit.runtime.scriptrunner.get_script_run_ctx', return_value=context), patch('streamlit.runtime.exists', return_value=True), patch('streamlit.runtime.get_instance', return_value=instance):
            reset_streamlit_session(state)
        context.uploaded_file_mgr.remove_session_files.assert_called_once_with('current-session')
        manager.clear_session_refs.assert_called_once_with('current-session')
        manager.remove_orphaned_files.assert_called_once()
        self.assertEqual(state, {'upload_generation': 1})
