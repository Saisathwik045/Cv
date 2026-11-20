"""Generated malformed/extreme images and session cycles; no checkpoint access."""
import importlib.util
import io
import struct
import unittest
import zlib
from unittest.mock import patch

from trustretina.app.state import clear_patient, reconcile_uploads


class SessionRobustness(unittest.TestCase):
    def test_twenty_replacements_and_resets_never_keep_stale_outputs(self):
        state = {}
        for cycle in range(20):
            left = f'generated-left-{cycle}'.encode()
            reconcile_uploads(state, left, b'generated-right')
            state.update(prediction='fixture', explanation='fixture', explanation_target='D')
            reconcile_uploads(state, left, b'generated-right')
            self.assertIn('prediction', state)
            reconcile_uploads(state, left, None)
            for key in ('prediction', 'explanation', 'explanation_target'):
                self.assertNotIn(key, state)
            clear_patient(state)
            self.assertEqual(state, {'upload_generation': cycle + 1})


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('torch', 'torchvision', 'PIL')),
                     'optional app CPU environment')
class ImageRobustness(unittest.TestCase):
    def encoded(self, size=(17, 11), mode='RGB', fmt='PNG', **kwargs):
        from PIL import Image
        stream = io.BytesIO()
        Image.new(mode, size).save(stream, format=fmt, **kwargs)
        return stream.getvalue()

    def test_truncated_valid_headers_and_unsupported_format_are_rejected(self):
        from trustretina.inference.service import decode_image
        for fmt in ('PNG', 'JPEG'):
            data = self.encoded(fmt=fmt)
            with self.subTest(fmt=fmt), self.assertRaises(ValueError):
                decode_image(data[:len(data)//2])
        with self.assertRaisesRegex(ValueError, 'JPEG or PNG'):
            decode_image(self.encoded(fmt='BMP'))

    def test_pixel_limit_is_checked_before_decoding_a_large_declared_image(self):
        from PIL import Image
        from trustretina.inference.service import decode_image
        data = bytearray(self.encoded(size=(1, 1)))
        # Valid PNG IHDR/CRC with declared dimensions just beyond 25 million.
        # No large bitmap is allocated and no IDAT payload is decoded.
        data[16:24] = struct.pack('>II', 5001, 5000)
        data[29:33] = struct.pack('>I', zlib.crc32(data[12:29]) & 0xffffffff)
        with patch.object(Image.Image, 'load', side_effect=AssertionError('Must reject before decode')):
            with self.assertRaisesRegex(ValueError, '25 million'):
                decode_image(bytes(data))

    def test_grayscale_alpha_and_exif_orientation_are_normalized(self):
        from PIL import Image
        from trustretina.inference.service import decode_image
        for mode in ('L', 'RGBA'):
            self.assertEqual(decode_image(self.encoded(mode=mode)).mode, 'RGB')
        exif = Image.Exif();exif[274] = 6
        rotated = decode_image(self.encoded(size=(9, 5), fmt='JPEG', exif=exif))
        self.assertEqual(rotated.size, (5, 9))
        self.assertEqual(rotated.mode, 'RGB')

    def test_extreme_aspect_pairs_prepare_finite_224_tensors_without_a_model(self):
        import torch
        from trustretina.inference.service import InferenceService
        from trustretina.preprocessing.shared import PreprocessConfig
        service = object.__new__(InferenceService)
        service.config = PreprocessConfig();service.device = torch.device('cpu')
        tensors, canvases = service.prepare(self.encoded(size=(1, 16384), mode='L'),
                                           self.encoded(size=(16384, 1), mode='RGBA'))
        self.assertEqual(tuple(tensors.shape), (2, 3, 224, 224))
        self.assertTrue(torch.isfinite(tensors).all())
        self.assertEqual([x.shape for x in canvases], [(224, 224, 3)] * 2)
        self.assertFalse(hasattr(service, 'model'))


if __name__ == '__main__':
    unittest.main()
