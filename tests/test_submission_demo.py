"""Synthetic Streamlit review: generated images and a fake service, no weights/data."""
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                         ('streamlit', 'torch', 'torchvision')), 'optional app CPU environment')
class SyntheticSubmissionDemoTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        from PIL import Image
        self.root = Path(__file__).resolve().parents[1]
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        content = io.BytesIO()
        Image.new('RGB', (32, 24), (90, 100, 110)).save(content, format='PNG')
        self.image = content.getvalue()
        other = io.BytesIO()
        Image.new('RGB', (32, 24), (110, 90, 100)).save(other, format='PNG')
        self.other = other.getvalue()
        codes = list('NDGCAHMO')
        scores = {k: .7 if k == 'D' else .1 for k in codes}
        self.service = Mock()
        self.service.predict.return_value = {
            'class_order': codes, 'positive_labels': ['D'],
            'raw_probabilities': scores, 'calibrated_probabilities': scores,
            'signals': {'calibrated': {'confidence': .6, 'uncertainty': .001, 'quality': .5}},
            'device': 'cpu synthetic fixture', 'inference_seconds': 0.0,
        }

    def app(self, contents, *, failure=None):
        from streamlit.testing.v1 import AppTest
        def upload(widget, label, *args, **kwargs):
            data = contents[0 if label == 'Left eye' else 1]
            return io.BytesIO(data) if data is not None else None
        self.upload_patch = patch('streamlit.delta_generator.DeltaGenerator.file_uploader', upload)
        self.upload_patch.start()
        self.addCleanup(self.upload_patch.stop)
        self.model_patch = patch('trustretina.inference.service.InferenceService',
                                 return_value=self.service, side_effect=failure)
        self.model_constructor = self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        return AppTest.from_file(str(self.root / 'streamlit_app.py')).run(timeout=30)

    def test_missing_or_invalid_eye_blocks_service(self):
        for contents in ((self.image, None), (self.image, b'invalid synthetic image')):
            with self.subTest(contents_kind='missing' if contents[1] is None else 'invalid'):
                app = self.app(contents)
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(next(b for b in app.button if b.label == 'Analyze paired images').disabled)
                self.model_constructor.assert_not_called()
                self.service.predict.assert_not_called()
                self.model_patch.stop()
                self.upload_patch.stop()

    def test_synthetic_pair_results_explanation_and_reset(self):
        import numpy as np
        self.service.explain.return_value = ({'zero_maps': [False, False]},
                                             [np.zeros((24, 32, 3), dtype=np.uint8)] * 2)
        app = self.app((self.image, self.other))
        app.checkbox[0].check().run(timeout=30)
        next(b for b in app.button if b.label == 'Analyze paired images').click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.service.predict.assert_called_once_with(self.image, self.other)
        self.assertFalse(any('No validated automatic acceptance policy' in w.value for w in app.warning))
        self.assertTrue(any('Diabetes (D)' in x.value for x in app.markdown))
        self.assertFalse(any(x.label == 'Download complete demonstration report' for x in app.get('download_button')))
        self.assertEqual(app.dataframe[0].value['Label'].tolist(), list('NDGCAHMO'))
        self.assertTrue(any(x.label == 'Download research result JSON' for x in app.get('download_button')))
        app.selectbox[0].set_value('H').run(timeout=30)
        next(b for b in app.button if b.label == 'Generate per-eye Grad-CAM').click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.service.explain.assert_called_once_with(self.image, self.other, 5)
        self.assertTrue(any(x.label == 'Download complete demonstration report' for x in app.get('download_button')))
        app.selectbox[0].set_value('D').run(timeout=30)
        self.assertFalse(any(x.label == 'Download complete demonstration report' for x in app.get('download_button')))
        next(b for b in app.button if b.label == 'Reset patient session').click().run(timeout=30)
        self.assertNotIn('prediction', app.session_state.filtered_state)
        self.assertNotIn('explanation', app.session_state.filtered_state)
        self.assertFalse(any(x.label == 'Download complete demonstration report' for x in app.get('download_button')))

    def test_duplicate_upload_and_missing_confirmation_block_prediction(self):
        app = self.app((self.image, self.image))
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('same image was uploaded twice' in e.value for e in app.error))
        self.assertTrue(next(b for b in app.button if b.label == 'Analyze paired images').disabled)
        self.service.predict.assert_not_called()
        self.model_patch.stop()
        self.upload_patch.stop()
        app = self.app((self.image, self.other))
        self.assertTrue(next(b for b in app.button if b.label == 'Analyze paired images').disabled)
        self.assertFalse(app.checkbox[0].value)

    def test_missing_bundle_error_is_visible_without_random_fallback(self):
        app = self.app((self.image, self.other), failure=FileNotFoundError('Synthetic bundle unavailable'))
        app.checkbox[0].check().run(timeout=30)
        next(b for b in app.button if b.label == 'Analyze paired images').click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('Prediction unavailable' in e.value for e in app.error))
        self.assertNotIn('prediction', app.session_state.filtered_state)
        self.service.predict.assert_not_called()


if __name__ == '__main__':
    unittest.main()
