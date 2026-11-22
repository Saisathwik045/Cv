"""Academic reasoning/state checks; no research metrics, models or private data."""
import importlib.util
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import Mock, patch

from dashboard.examiner import SCENARIOS, feedback, rehearsal_export

ROOT = Path(__file__).resolve().parents[1]


class ExaminerContracts(unittest.TestCase):
    def test_partial_invalid_and_correct_answers_are_not_invented_grades(self):
        self.assertTrue(all(row['status'] == 'UNANSWERED' for row in feedback({})))
        for invalid in (True, -1, 99, '0', None):
            self.assertEqual(feedback({'labels': invalid})[0]['status'], 'UNANSWERED')
        correct = {s['id']: s['correct'] for s in SCENARIOS}
        self.assertTrue(all(row['status'] == 'ALIGNED' for row in feedback(correct)))
        wrong = {s['id']: 1 - s['correct'] for s in SCENARIOS}
        self.assertTrue(all(row['status'] == 'REVISIT' for row in feedback(wrong)))
        output = rehearsal_export(correct)
        self.assertFalse(output['research_evaluation'])
        self.assertFalse(output['clinical_validation'])
        self.assertNotIn('score', output)
        self.assertNotIn('answers', output)
        self.assertEqual(output['decision'], 'REVIEW REQUIRED')
        self.assertEqual(json.loads(json.dumps(output)), output)

    def test_teaching_evidence_and_guide_resolve_without_private_paths(self):
        self.assertEqual(len({s['id'] for s in SCENARIOS}), len(SCENARIOS))
        for item in SCENARIOS:
            self.assertTrue((ROOT / item['evidence']).is_file())
            self.assertIn(item['correct'], range(len(item['options'])))
        guide = ROOT / 'docs/EXAMINER_GUIDE.md'
        text = guide.read_text()
        self.assertNotRegex(text, r'/(workspace|home|tmp|kaggle)/|(?i:x-amz-signature=)')
        for link in re.findall(r'\]\(([^)]+)\)', text):
            self.assertTrue((guide.parent / link).is_file(), link)
        self.assertIn('not an official institutional', text)


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'optional Streamlit')
class ExaminerPageTests(unittest.TestCase):
    def test_interactive_feedback_updates_and_reset_is_scoped_without_model_access(self):
        from streamlit.testing.v1 import AppTest
        service = Mock()
        service.InferenceService.side_effect = AssertionError('No private model access')
        with patch.dict(sys.modules, {'trustretina.inference.service': service}):
            app = AppTest.from_string('from dashboard.examiner import render_examiner_page\nrender_examiner_page()\n').run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.radio), 6)  # Six standalone scenarios.
            self.assertTrue(any('Not a grade' in w.value for w in app.warning))
            self.assertFalse(app.expander)
            next(b for b in app.button if b.label == 'Check reasoning').click().run()
            self.assertTrue(all('UNANSWERED' in e.label for e in app.expander))
            app.radio(key='examiner_labels').set_value(SCENARIOS[0]['options'][1]).run()
            self.assertIn('REVISIT', app.expander[0].label)
            app.radio(key='examiner_labels').set_value(SCENARIOS[0]['options'][0]).run()
            self.assertIn('ALIGNED', app.expander[0].label)
            self.assertEqual(len(app.get('download_button')), 2)
            app.session_state['guided_replacement'] = True
            next(b for b in app.button if b.label == 'Reset rehearsal').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.expander)
            self.assertIsNone(app.radio(key='examiner_labels').value)
            self.assertTrue(app.session_state['guided_replacement'])
            service.InferenceService.assert_not_called()


if __name__ == '__main__':
    unittest.main()
