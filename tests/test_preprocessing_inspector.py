"""Exact synthetic preprocessing and no-model UI contracts."""
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from dashboard import inspector

ROOT = Path(__file__).resolve().parents[1]
HAS_CPU = all(importlib.util.find_spec(n) for n in ('torch','torchvision','PIL'))


@unittest.skipUnless(HAS_CPU, 'optional app CPU environment')
class InspectorContracts(unittest.TestCase):
    def test_presets_match_actual_service_preparation_without_loading_weights(self):
        import numpy as np
        import torch
        from trustretina.inference.service import InferenceService
        from trustretina.preprocessing.shared import PreprocessConfig
        service = object.__new__(InferenceService)
        service.config = PreprocessConfig();service.device = torch.device('cpu')
        expectations = [([224,168],[0,28,0,28]), ([168,224],[28,0,28,0]),
                        ([224,224],[0,0,0,0]), ([224,28],[0,98,0,98])]
        with patch('torch.load', side_effect=AssertionError('No checkpoint loading')):
            for shape, (size, padding) in zip(inspector.SHAPES, expectations):
                content = inspector.fixture(shape, 'Left')
                data = inspector.inspect(content)
                tensors, canvases = service.prepare(content, content)
                self.assertEqual(data['geometry']['resized_size'], size)
                self.assertEqual(data['geometry']['padding'], padding)
                self.assertEqual(data['tensor_shape'], [3,224,224])
                np.testing.assert_array_equal(np.asarray(data['canvas']), canvases[0])
                for idx, stats in enumerate(data['channel_statistics']):
                    self.assertAlmostEqual(stats['Minimum'], float(tensors[0,idx].min()), places=6)
                    self.assertAlmostEqual(stats['Maximum'], float(tensors[0,idx].max()), places=6)
                overlay = inspector.boundary_overlay(data['canvas'], data['geometry'])
                self.assertIsNot(overlay, data['canvas'])
                np.testing.assert_array_equal(np.asarray(data['canvas']), canvases[0])
        self.assertFalse(hasattr(service, 'model'))

    def test_padding_bytes_and_nonzero_normalized_padding_are_reported_honestly(self):
        data = inspector.inspect(inspector.fixture(next(iter(inspector.SHAPES)), 'Right'))
        self.assertEqual(data['padding_rgb'], [124,116,104])
        self.assertEqual(data['canvas'].getpixel((0,0)), (124,116,104))
        self.assertAlmostEqual(data['padding_fraction'], .25)
        self.assertEqual(data['source'].size, (320,240))

    def test_decode_orientation_invalid_input_and_identity_refusal(self):
        from PIL import Image
        exif=Image.Exif();exif[274]=6
        stream=io.BytesIO();Image.new('RGB',(9,5)).save(stream,format='JPEG',exif=exif)
        data=inspector.inspect(stream.getvalue())
        self.assertEqual(data['geometry']['original_size'], [5,9])
        with self.assertRaises(ValueError):inspector.inspect(b'not an image')
        with patch('trustretina.inference.bundle.verified_contract',return_value={'preprocessing_sha256':'wrong'}):
            with self.assertRaisesRegex(ValueError,'identity'):inspector.inspect(b'')
        with self.assertRaises(ValueError):inspector.fixture('arbitrary geometry', 'Left')
        with self.assertRaises(ValueError):inspector.fixture(next(iter(inspector.SHAPES)), 'unknown')


@unittest.skipUnless(HAS_CPU and importlib.util.find_spec('streamlit'), 'optional app UI environment')
class InspectorPage(unittest.TestCase):
    def test_navigation_controls_overlay_reset_and_failure_without_model(self):
        from streamlit.testing.v1 import AppTest
        # The real preprocessing imports the service decoder. Patch only its constructor.
        with patch('trustretina.inference.service.InferenceService',side_effect=AssertionError('No model constructor')) as ctor, \
             patch('torch.load',side_effect=AssertionError('No checkpoint loading')):
            app=AppTest.from_file(str(ROOT/'streamlit_app.py')).run(timeout=30)
            app.sidebar.radio[0].set_value('Preprocessing inspector').run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(any('SYNTHETIC NON-PATIENT' in w.value for w in app.warning))
            self.assertEqual(app.dataframe[0].value.iloc[1]['Width'],224)
            self.assertEqual(app.metric[0].value,'25.00%')
            app.selectbox[0].set_value('Square · 320 × 320').run()
            self.assertEqual(app.metric[0].value,'0.00%')
            app.checkbox[0].check().run()
            app.selectbox[1].set_value('Right').run()
            self.assertFalse(app.exception)
            self.assertFalse(app.get('download_button'))
            app.session_state['guided_replacement']=True
            next(b for b in app.button if b.label=='Reset inspector').click().run()
            self.assertEqual(app.selectbox[0].value,next(iter(inspector.SHAPES)))
            self.assertEqual(app.selectbox[0].key,'inspector_shape_1')
            self.assertFalse(app.checkbox[0].value)
            self.assertTrue(app.session_state['guided_replacement'])
            with patch('dashboard.inspector.inspect',side_effect=ValueError('secret fixture value')):
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(any('No substitute canvas' in x.value for x in app.error))
                self.assertFalse(app.dataframe)
                self.assertNotIn('secret fixture value',str([x.value for x in app.error]))
            ctor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
