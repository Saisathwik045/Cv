"""Public reporting contracts; never require models, private rows or fitting."""
import importlib.util
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from dashboard.academic import (
    ROOT, CLASSES, load_academic_evidence, metric_rows,
    class_rows, risk_rows,
)


class AcademicEvidenceTests(unittest.TestCase):
    def test_recorded_values_preserve_cohorts_classes_and_negative_findings(self):
        data = load_academic_evidence()
        self.assertEqual(len(metric_rows(data)), 8)
        for role in ('calibration', 'validation'):
            rows = class_rows(data, role)
            self.assertEqual([r['Class'] for r in rows], [c for c in CLASSES for _ in range(2)])
            self.assertEqual(len(risk_rows(data, role)), 12)
            for run in data['runs'].values():
                recorded = run['roles'][role]
                for c in recorded['per_class_classification']:
                    row = next(r for r in rows if r['Class'] == c['label'] and r['Resolution'] == run['resolution'])
                    self.assertEqual((row['FP'], row['FN'], row['Positive']), (c['fp'], c['fn'], c['positive']))
                    self.assertEqual(row['F1'], c['f1'] if c['f1_defined'] else None)
                self.assertTrue(recorded['binary_decisions_unchanged'])
                self.assertGreater(recorded['calibrated']['metrics']['macro_ece_10_equal_width_bins'], recorded['raw']['metrics']['macro_ece_10_equal_width_bins'])

    def test_missing_or_tampered_evidence_and_figures_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(OSError):
                load_academic_evidence(root)
            source = ROOT / 'results/calibration-v15'
            target = root / 'results/calibration-v15'
            shutil.copytree(source, target)
            (target / 'aggregate-results.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_academic_evidence(root)
            shutil.copyfile(source / 'aggregate-results.json', target / 'aggregate-results.json')
            next((target / 'figures').glob('*.png')).write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'figure checksum'):
                load_academic_evidence(root)

    def test_comparison_document_contains_exact_recorded_class_rows(self):
        text = (ROOT / 'docs/ALL_EIGHT_CLASS_COMPARISON.md').read_text()
        for row in class_rows(load_academic_evidence(), 'validation'):
            cells = [row['Class'], str(row['Resolution']), str(row['Positive']), str(row['Negative'])]
            cells += ['undefined' if row[k] is None else f'{row[k]:.6f}' for k in ('Precision', 'Recall', 'F1')]
            cells += [str(row['FP']), str(row['FN']), f"{row['Raw ECE']:.6f}", f"{row['Calibrated ECE']:.6f}"]
            self.assertIn('| ' + ' | '.join(cells) + ' |', text)

    def test_new_documents_have_resolving_links_and_no_private_paths(self):
        for name in ('ACADEMIC_RESEARCH_DEFENCE.md', 'ALL_EIGHT_CLASS_COMPARISON.md'):
            p = ROOT / 'docs' / name
            text = p.read_text()
            self.assertNotRegex(text, r'/(workspace|tmp|home|kaggle)/|(?i:x-amz-signature=|x-goog-signature=)')
            for link in re.findall(r'\]\(([^)]+)\)', text):
                self.assertTrue((p.parent / link).is_file(), link)


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'optional Streamlit')
class AcademicPageTests(unittest.TestCase):
    def test_public_page_never_initializes_private_service_and_fails_closed(self):
        from streamlit.testing.v1 import AppTest
        service = Mock()
        service.InferenceService.side_effect = AssertionError('No model access')
        with patch.dict(sys.modules, {'trustretina.inference.service': service}):
            app = AppTest.from_string('from dashboard.academic import render_academic_page\nrender_academic_page()\n').run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.tabs), 5)
            self.assertTrue(any('REVIEW REQUIRED' in w.value for w in app.warning))
            self.assertTrue(any('original v5' in i.value for i in app.info))
            self.assertEqual(app.dataframe[3].value['Class'].tolist(), [c for c in CLASSES for _ in range(2)])
            app.selectbox[0].set_value('calibration').run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.dataframe[3].value.iloc[0]['Positive'], 116)
            service.InferenceService.assert_not_called()
            with patch('dashboard.academic.load_academic_evidence', side_effect=ValueError('tampered')):
                app.run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any('No substitute' in e.value for e in app.error))
                self.assertEqual(len(app.dataframe), 0)


if __name__ == '__main__':
    unittest.main()
