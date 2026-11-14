"""Public aggregate/report consistency only; no private input or fitting access."""
import hashlib
import json
import re
import struct
from pathlib import Path
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/calibration-v15'
RUNS = ('v14_candidate_res320_seed2026', 'v14_reference_res224_seed2026')
CLASSES = list('NDGCAHMO')


class PublicV15ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((BASE / 'aggregate-results.json').read_text())
        cls.report = (BASE / 'RESULTS.md').read_text()

    def test_public_schema_rejects_private_record_containers(self):
        self.assertEqual(set(self.data), {
            'schema', 'data_kind', 'class_order', 'deployment_default',
            'independent_validation_complete', 'cohorts', 'provenance', 'runs',
            'paired_hamming_aurc_intervals', 'figures'})
        self.assertEqual(self.data['schema'], 'v15_public_aggregate_results_v1')
        forbidden = {'patient_id', 'patient_ids', 'component_id', 'component_ids',
                     'image_ids', 'image_id', 'rows', 'labels', 'probabilities',
                     'signals', 'seeds', 'passes', 'approval', 'user_approval_reference',
                     'sorted_training_quality_distributions', 'path', 'directory'}

        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)
            elif isinstance(value, str):
                self.assertNotRegex(value, r'/workspace/|/tmp/|/home/|/kaggle/|(?i:signature=|x-amz-|x-goog-|token=)')

        inspect(self.data)

    def test_exact_models_and_frozen_protocol_are_bound(self):
        registry_path = ROOT / 'results/candidate-v14/verified-artifact-registry.json'
        registry = json.loads(registry_path.read_text())
        provenance = self.data['provenance']
        for name, path in [('registry_sha256', registry_path),
                           ('frozen_plan_sha256', ROOT / 'results/candidate-v14/frozen-plan.json'),
                           ('protocol_sha256', ROOT / 'docs/CALIBRATION_RELIABILITY_V15.md')]:
            self.assertEqual(provenance[name], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(tuple(self.data['runs']), RUNS)
        self.assertEqual(self.data['class_order'], CLASSES)
        for fit in registry['fits']:
            run = self.data['runs'][fit['run_id']]
            self.assertEqual(run['kernel_version'], fit['kernel_version'])
            self.assertEqual(run['best_checkpoint_sha256'], fit['artifact_sha256']['best.pt'])

    def test_all_cohorts_classes_and_supports_are_disclosed(self):
        self.assertEqual(self.data['cohorts']['train_quality'], {'patients': 2275, 'eyes': 4550})
        for role, n, groups in [('calibration', 350, 331), ('validation', 525, 518)]:
            self.assertEqual(self.data['cohorts'][role], {'patients': n, 'components': groups, 'mc_passes': 20})
            for run in self.data['runs'].values():
                d = run['roles'][role]
                self.assertEqual([r['label'] for r in d['per_class_classification']], CLASSES)
                for mode in ('raw', 'calibrated'):
                    classes = d[mode]['per_class_reliability']
                    self.assertEqual([r['label'] for r in classes], CLASSES)
                    self.assertTrue(all(r['positive'] + r['negative'] == n for r in classes))
                    self.assertAlmostEqual(sum(r['ece'] for r in classes) / 8,
                                           d[mode]['metrics']['macro_ece_10_equal_width_bins'])

    def test_optimizer_and_unchanged_classification_evidence(self):
        for run in self.data['runs'].values():
            fit = run['fit']
            self.assertTrue(fit['optimizer_success'])
            self.assertFalse(fit['boundary_hit'])
            self.assertGreater(fit['temperature'], .05)
            self.assertLess(fit['temperature'], 20)
            self.assertIn(f"{fit['temperature']:.12f}", self.report)
            for d in run['roles'].values():
                self.assertTrue(d['binary_decisions_unchanged'])
                for c in d['per_class_classification']:
                    self.assertEqual(c['tp'] + c['fn'], c['positive'])
                    self.assertEqual(c['tn'] + c['fp'], c['negative'])
                    self.assertAlmostEqual(c['f1'], 2*c['tp'] / (2*c['tp'] + c['fp'] + c['fn']))

    def test_negative_ece_and_quality_findings_are_preserved(self):
        for run in self.data['runs'].values():
            for d in run['roles'].values():
                self.assertGreater(d['calibrated']['metrics']['macro_ece_10_equal_width_bins'],
                                   d['raw']['metrics']['macro_ece_10_equal_width_bins'])
                for mode in ('raw', 'calibrated'):
                    curves = d[mode]['risk_coverage']
                    self.assertGreater(curves['confidence_uncertainty_quality']['hamming_aurc'],
                                       curves['confidence_uncertainty']['hamming_aurc'])
                    self.assertGreater(curves['confidence_uncertainty_quality']['hamming_aurc'],
                                       curves['confidence']['hamming_aurc'])

    def test_paired_intervals_are_component_descriptive_only(self):
        for modes in self.data['paired_hamming_aurc_intervals'].values():
            for strategies in modes.values():
                for d in strategies.values():
                    self.assertEqual((d['replicates'], d['seed']), (200, 2026))
                    self.assertEqual(d['unit'], 'whole containment component')
                    low, high = d['320_minus_224_hamming_aurc_percentile_95']
                    self.assertLessEqual(low, 0)
                    self.assertGreaterEqual(high, 0)
        self.assertFalse(self.data['independent_validation_complete'])
        self.assertEqual(self.data['deployment_default'], 'REVIEW REQUIRED')

    def test_report_cohort_cells_match_recorded_aggregates(self):
        keys = ('patient_nll', 'component_equal_nll', 'brier', 'component_equal_brier',
                'macro_ece_10_equal_width_bins')
        for run in self.data['runs'].values():
            for role, d in run['roles'].items():
                cells = [f"{d['raw']['metrics'][k]:.6f} → {d['calibrated']['metrics'][k]:.6f}" for k in keys]
                self.assertIn('| ' + ' | '.join([role, str(run['resolution'])] + cells) + ' |', self.report)

    def test_reviewed_png_bytes_and_metadata_are_bound(self):
        self.assertEqual(len(self.data['figures']), 4)
        for name, info in self.data['figures'].items():
            blob = (BASE / name).read_bytes()
            self.assertEqual(hashlib.sha256(blob).hexdigest(), info['sha256'])
            self.assertEqual(info['sha256'], info['source_output_sha256'])
            self.assertEqual(blob[:8], b'\x89PNG\r\n\x1a\n')
            pos = 8
            while pos < len(blob):
                n = struct.unpack('>I', blob[pos:pos+4])[0]
                kind, payload = blob[pos+4:pos+8], blob[pos+8:pos+8+n]
                self.assertNotIn(kind, (b'eXIf', b'iTXt', b'zTXt'))
                if kind == b'tEXt':
                    key, text = payload.split(b'\0', 1)
                    self.assertEqual(key, b'Software')
                    self.assertTrue(text.startswith(b'Matplotlib version'))
                self.assertEqual(zlib.crc32(kind + payload) & 0xffffffff,
                                 struct.unpack('>I', blob[pos+8+n:pos+12+n])[0])
                pos += n + 12
            self.assertEqual(pos, len(blob))

    def test_report_links_and_current_status_agree(self):
        for url in re.findall(r'\]\(([^)]+)\)', self.report):
            self.assertFalse(url.startswith('/'))
            self.assertTrue((BASE / url).resolve().is_file(), url)
        for relative in ('README.md', 'docs/STATUS.md'):
            text = (ROOT / relative).read_text()
            self.assertIn('results/calibration-v15/RESULTS.md', text)
            self.assertIn('1.105496125613', text)
            self.assertIn('1.002532561130', text)
            self.assertIn('Independent validation', text)
        self.assertNotRegex(self.report, r'/workspace/|/tmp/|/home/|/kaggle/|(?i:signature=|x-amz-|x-goog-|token=)')


if __name__ == '__main__':
    unittest.main()
