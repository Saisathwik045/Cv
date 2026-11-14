"""Public academic document/source checks; no private cohort or fitting access."""
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ('FINAL_RESEARCH_REPORT.md', 'PHASE_COMPLETION_AUDIT.md',
        'FINAL_PRESENTATION_OUTLINE.md', 'FINAL_SUBMISSION_CHECKLIST.md')


class AcademicSubmissionDocsTests(unittest.TestCase):
    def test_public_sources_are_hash_bound_without_private_access(self):
        audit = json.loads((ROOT / 'results/submission-audit/source-provenance.json').read_text())
        self.assertFalse(audit['private_patient_data_accessed'])
        self.assertFalse(audit['historical_final_cohort_accessed'])
        self.assertFalse(audit['new_research_fitting_or_inference'])
        self.assertFalse(audit['independent_validation_complete'])
        for relative, digest in audit['sources'].items():
            self.assertFalse(relative.startswith('/'))
            self.assertNotIn('..', Path(relative).parts)
            self.assertNotIn('results/final-v7/', relative)
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest)

    def test_phase_audit_covers_zero_through_fifteen_and_boundaries(self):
        text = (ROOT / 'docs/PHASE_COMPLETION_AUDIT.md').read_text()
        self.assertEqual([int(n) for n in re.findall(r'^\| (\d+) \|', text, re.M)], list(range(16)))
        for phrase in ('Independent', 'native Mac', 'backup', 'REVIEW REQUIRED'):
            # Deployment default can be stated in the linked final report.
            self.assertIn(phrase.lower(), (text + (ROOT / 'docs/FINAL_RESEARCH_REPORT.md').read_text()).lower())

    def test_report_tables_match_existing_public_values(self):
        text = (ROOT / 'docs/FINAL_RESEARCH_REPORT.md').read_text()
        v5 = json.loads((ROOT / 'results/full-v5/comparison.json').read_text())
        for row in v5['comparison']:
            cells = [row['backbone'], f"{row['epochs']} / {row['best_epoch']}"]
            cells += [f"{row[k]:.4f}" for k in ('patient_macro_auroc', 'patient_macro_auprc',
                                               'patient_macro_f1', 'patient_micro_f1')]
            self.assertIn('| ' + ' | '.join(cells) + ' |', text)
        v15 = json.loads((ROOT / 'results/calibration-v15/aggregate-results.json').read_text())
        for run in v15['runs'].values():
            self.assertIn(f"{run['fit']['temperature']:.12f}", text)
            for role, row in run['roles'].items():
                cells = [role, str(run['resolution'])] + [
                    f"{row['raw']['metrics'][k]:.6f} → {row['calibrated']['metrics'][k]:.6f}"
                    for k in ('patient_nll', 'brier', 'macro_ece_10_equal_width_bins')]
                self.assertIn('| ' + ' | '.join(cells) + ' |', text)
        for phrase in ('ECE increased', 'quality worsened', 'Independent validation is incomplete',
                       'fake service', '0.722602', 'REVIEW REQUIRED'):
            self.assertIn(phrase, text)

    def test_submission_links_privacy_and_mac_instructions(self):
        for name in DOCS:
            text = (ROOT / 'docs' / name).read_text()
            self.assertNotRegex(text, r'/(?:workspace|tmp|home|kaggle)/|(?i:x-amz-signature=|x-goog-signature=)')
            for link in re.findall(r'\]\(([^)]+)\)', text):
                if link.startswith('https://'):
                    continue
                self.assertTrue((ROOT / 'docs' / link).resolve().is_file(), link)
        mac = (ROOT / 'docs/MACOS_QUICKSTART.md').read_text()
        self.assertIn('git clone --branch main ', mac)
        self.assertNotIn('git clone --branch phase/08-macos-launcher', mac)
        self.assertIn('not a native installation/runtime test', mac.lower().replace('*', ''))


if __name__ == '__main__':
    unittest.main()
