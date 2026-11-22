"""Offline export completeness, escaping and current-target behaviour on synthetic inputs."""
import importlib.util
import unittest
from html.parser import HTMLParser


@unittest.skipUnless(importlib.util.find_spec('PIL'), 'optional image dependency')
class DemonstrationReportTests(unittest.TestCase):
    def record(self):
        labels = list('NDGCAHMO')
        return {'class_order': labels, 'positive_labels': ['D'],
                'raw_probabilities': dict.fromkeys(labels, .7),
                'calibrated_probabilities': dict.fromkeys(labels, .8),
                'signals': {'calibrated': {'confidence': .8, 'uncertainty': .003, 'quality': .6}},
                'threshold': .5, 'device': 'SYNTHETIC NON-PATIENT TEST',
                'limitations': ['</pre><script>alert("fixture")</script>']}

    def test_export_embeds_four_images_and_full_record_without_external_resources(self):
        from PIL import Image
        import base64
        import io
        from dashboard.report import demonstration_report
        image = Image.new('RGB', (24, 16), (90, 60, 40))
        metadata = Image.Exif(); metadata[0x010E] = 'PRIVATE METADATA FIXTURE'
        image.info['exif'] = metadata.tobytes()
        report = demonstration_report(self.record(), [image, image], [image, image], 'D', zero_maps=[True, False])
        class Tags(HTMLParser):
            def __init__(self):
                super().__init__(); self.images = []; self.scripts = []
            def handle_starttag(self, tag, attrs):
                if tag == 'img': self.images.append(dict(attrs)['src'])
                if tag == 'script': self.scripts.append(attrs)
        tags = Tags(); tags.feed(report)
        self.assertEqual(len(tags.images), 4)
        self.assertTrue(all(src.startswith('data:image/png;base64,') for src in tags.images))
        for src in tags.images:
            with Image.open(io.BytesIO(base64.b64decode(src.split(',', 1)[1]))) as preview:
                self.assertFalse(preview.getexif())
                self.assertNotIn('exif', preview.info)
        self.assertFalse(tags.scripts)
        self.assertIn('&lt;script&gt;', report)
        self.assertIn('SYNTHETIC NON-PATIENT TEST', report)
        self.assertIn('Diabetes (D)', report)
        self.assertIn('zero activation map', report)
        self.assertIn('Complete exported prediction record', report)
        self.assertIn('not accuracy scores', report)
        self.assertEqual(report.count('<rect '), 16)
        self.assertIn("default-src 'none'", report)

    def test_incomplete_pair_or_unrecognized_target_cannot_be_a_complete_report(self):
        from PIL import Image
        from dashboard.report import demonstration_report
        image = Image.new('RGB', (5, 5))
        for inputs, maps, target in [([image], [image, image], 'D'),
                                     ([image, image], [image], 'D'),
                                     ([image, image], [image, image], 'unknown')]:
            with self.subTest(target=target, inputs=len(inputs), maps=len(maps)), self.assertRaises(ValueError):
                demonstration_report(self.record(), inputs, maps, target)
