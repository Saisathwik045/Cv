"""V15 synthetic-only fixtures; no real patients, checkpoint loading or network."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

import numpy as np

from trustretina.config import CLASSES
from trustretina.reliability.development import fit_shared_temperature, temperature_probabilities
from trustretina_v15.v15 import analyze, descriptive, diagnostic, save_report, write_json
from trustretina_v15.v15_fixture import create_fixture
from trustretina_v15.v15_inputs import (
    FIXTURE, RUNS, audit_roles, load_bundle, load_npz, sha256, verified_path,
    validate_original_role,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'a'*64


class V15Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path, self.digest, self.approval = create_fixture(self.root/'input', PROTOCOL)
        self.manifest = json.loads(self.path.read_text())

    def update(self):
        write_json(self.path, self.manifest)
        self.digest = sha256(self.path)
        self.approval['manifest_sha256'] = self.digest

    def load(self, approval=None):
        return load_bundle(self.path, self.digest, PROTOCOL,
                           self.approval if approval is None else approval, fixture=True)

    def change_artifact(self, run, name, **arrays):
        path = self.path.parent/run/name
        np.savez_compressed(path, **arrays)
        fit = next(f for f in self.manifest['fixture_registry']['fits'] if f['run_id'] == run)
        fit['artifact_sha256'][name] = sha256(path)
        stage_path = path.parent/'stage-report.json'
        stage = json.loads(stage_path.read_text())
        stage['output_hashes'] = fit['artifact_sha256']
        write_json(stage_path, stage)
        fit['report_sha256'] = sha256(stage_path)
        self.update()

    def change_rows(self, role, transform):
        item = self.manifest['roles'][role]; path = self.path.parent/item['file']
        rows = json.loads(path.read_text()); transform(rows); write_json(path, rows)
        item['sha256'] = sha256(path); self.update()

    def test_unauthorized_fit_denied_before_any_input_read(self):
        with patch('trustretina_v15.v15_inputs.read_json') as read:
            with self.assertRaises(PermissionError):
                load_bundle('/missing/private/manifest.json', self.digest, PROTOCOL, None)
            read.assert_not_called()
        for field in ('approved', 'scope', 'manifest_sha256', 'protocol_sha256', 'data_kind', 'run_ids'):
            with self.subTest(field=field):
                approval = copy.deepcopy(self.approval); approval[field] = None
                with self.assertRaises(PermissionError):
                    self.load(approval)

    def test_fixture_permission_cannot_authorize_real_fitting(self):
        with self.assertRaises(PermissionError):
            load_bundle(self.path, self.digest, PROTOCOL, self.approval)
        approval = {**self.approval, 'data_kind': 'real_v14'}
        with self.assertRaises(PermissionError):
            load_bundle(self.path, self.digest, PROTOCOL, approval)

    def test_manifest_hash_verified_before_parsing(self):
        self.path.write_text(self.path.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.load()

    def test_checkpoint_hash_verified_as_opaque_bytes_without_torch(self):
        (self.path.parent/RUNS[1]/'best.pt').write_bytes(b'wrong')
        with patch('trustretina_v15.v15.fit_shared_temperature') as fit:
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                analyze(self.path, self.digest, PROTOCOL, self.approval, fixture=True)
            fit.assert_not_called()

    def test_every_recorded_artifact_hash_is_checked(self):
        # Not just the checkpoint: metadata, predictions, seeds and quality.
        for run in RUNS:
            for name in self.manifest['fixture_registry']['fits'][RUNS.index(run)]['artifact_sha256']:
                path = self.path.parent/run/name; before = path.read_bytes()
                with self.subTest(run=run, name=name):
                    path.write_bytes(before+b'corrupted')
                    with self.assertRaisesRegex(ValueError, 'SHA256'):
                        self.load()
                    path.write_bytes(before)

    def test_completed_stage_report_hash_and_provenance_are_required(self):
        path = self.path.parent/RUNS[0]/'stage-report.json'
        before = path.read_text(); path.write_text(before+' ')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.load()
        stage = json.loads(before); stage['calibration_inference_after_freeze'] = False
        write_json(path, stage)
        self.manifest['fixture_registry']['fits'][0]['report_sha256'] = sha256(path)
        self.update()
        with self.assertRaisesRegex(ValueError, 'report provenance'):
            self.load()

    def test_path_traversal_absolute_and_symlink_rejected(self):
        outside = self.root/'outside'; outside.write_bytes(b'opaque')
        for name in ('../outside', str(outside)):
            with self.assertRaises(ValueError):
                verified_path(self.path.parent, name, sha256(outside))
        (self.path.parent/'link').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            verified_path(self.path.parent, 'link', sha256(outside))
        (self.path.parent/'dirlink').symlink_to(outside.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            verified_path(self.path.parent, 'dirlink/outside', sha256(outside))

    def test_class_order_and_historical_final_role_rejected(self):
        original = copy.deepcopy(self.manifest)
        for change in ('class_order', 'roles'):
            self.manifest = copy.deepcopy(original)
            if change == 'class_order':
                self.manifest[change] = list(reversed(CLASSES))
            else:
                self.manifest['roles']['final_evaluation'] = {}
            self.update()
            with self.assertRaises(ValueError):
                self.load()

    def test_parent_manifest_version_and_freeze_checks(self):
        original = copy.deepcopy(self.manifest)
        for change in ('parent', 'version', 'freeze', 'matrix'):
            self.manifest = copy.deepcopy(original)
            if change == 'parent':
                self.manifest['roles']['calibration']['parent_manifest_sha256'] = '0'*64
            elif change == 'version':
                self.manifest['runs'][RUNS[0]]['kernel_version'] = 'wrong/2'
            elif change == 'matrix':
                self.manifest['fixture_registry']['fits'].reverse()
            else:
                self.manifest['fixture_registry']['fits'][0]['strict_best_reload'] = False
            self.update()
            with self.assertRaises(ValueError):
                self.load()

    def test_missing_eye_duplicate_eye_bad_labels_and_blank_identity_rejected(self):
        item = self.manifest['roles']['calibration']; path = self.path.parent/item['file']
        original = json.loads(path.read_text())
        for mutation in ('missing', 'duplicate', 'labels', 'blank', 'nonbinary', 'row_role'):
            rows = copy.deepcopy(original)
            if mutation == 'missing':
                rows.pop()
            elif mutation == 'duplicate':
                rows[1]['eye'] = rows[0]['eye']
            elif mutation == 'labels':
                rows[0]['labels'][0] = 1-rows[0]['labels'][0]
            elif mutation == 'blank':
                rows[0]['component_id'] = ''
            elif mutation == 'row_role':
                rows[0]['role'] = 'final_evaluation'
            else:
                rows[0]['labels'][0] = -1
            with self.subTest(mutation=mutation):
                write_json(path, rows); item['sha256'] = sha256(path); self.update()
                with self.assertRaises(ValueError):
                    self.load()

    def test_component_leakage_and_train_labels_rejected(self):
        roles = {r: json.loads((self.path.parent/x['file']).read_text()) for r, x in self.manifest['roles'].items()}
        roles['validation'][0]['component_id'] = roles['train'][0]['component_id']
        roles['validation'][1]['component_id'] = roles['train'][0]['component_id']
        with self.assertRaisesRegex(ValueError, 'crosses roles'):
            audit_roles(roles, fixture=True)
        roles['validation'][0]['component_id'] = roles['validation'][1]['component_id'] = 'fixture-valid'
        roles['train'][0]['labels'] = [0]*8
        with self.assertRaisesRegex(ValueError, 'label-free'):
            audit_roles(roles, fixture=True)

    def test_original_v4_object_schema_labels_and_eye_identity(self):
        rows = json.loads((self.path.parent/'calibration-rows.json').read_text())
        original = {'schema': 'phase2_eye_rows_v1', 'role': 'calibration', 'rows': rows}
        validate_original_role('calibration', original, copy.deepcopy(rows))
        for field in ('path', 'component_id', 'labels', 'eye'):
            changed = copy.deepcopy(rows)
            changed[0][field] = [0]*8 if field == 'labels' else 'wrong'
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_original_role('calibration', original, changed)
        with self.assertRaises(ValueError):
            validate_original_role('validation', original, rows)
        train = json.loads((self.path.parent/'train-rows.json').read_text())
        labeled = [{**r, 'labels': [0]*8} for r in train]
        validate_original_role('train', {'schema': 'phase2_eye_rows_v1', 'role': 'train', 'rows': labeled}, train)

    def test_mixed_component_labels_preserved_and_component_equal_nll(self):
        from trustretina.reliability.development import nll, component_weights
        roles = {r: json.loads((self.path.parent/x['file']).read_text()) for r, x in self.manifest['roles'].items()}
        # Two differently labeled patients may belong to a contained group.
        for row in roles['calibration'][2:4]:
            row['component_id'] = roles['calibration'][0]['component_id']
        audit_roles(roles, fixture=True)
        y = np.asarray([[0]*8, [1]*8, [1]*8]); p = np.asarray([[.2]*8, [.6]*8, [.8]*8])
        weights = component_weights(['shared', 'shared', 'single'])
        np.testing.assert_equal(weights, [.5, .5, 1.])
        expected = (-.5*np.log(.8)-.5*np.log(.6)-np.log(.8))/2
        self.assertAlmostEqual(nll(y, p, weights), expected)

    def test_sample_order_component_and_probability_aggregation_rejected(self):
        run = RUNS[0]; name = 'validation-patients.npz'
        original = load_npz(self.path.parent/run/name)
        for key in ('patient_ids', 'component_ids', 'probabilities'):
            changed = copy.deepcopy(original); changed[key] = changed[key][::-1]
            if np.array_equal(changed[key], original[key]):
                changed[key][0] = .7
            with self.subTest(key=key):
                self.change_artifact(run, name, **changed)
                with self.assertRaisesRegex(ValueError, 'aggregation/alignment'):
                    self.load()
        self.change_artifact(run, name, **original)

    def test_mc_requires_exact_twenty_passes_seeds_and_patient_alignment(self):
        run = RUNS[1]; name = 'calibration-passes.npz'
        original = load_npz(self.path.parent/run/name)
        for mutation in ('shape', 'seed', 'ids', 'nan', 'range'):
            changed = copy.deepcopy(original)
            if mutation == 'shape':
                changed['probabilities'] = changed['probabilities'][:19]
            elif mutation == 'seed':
                changed['seeds'][0] += 1
            elif mutation == 'ids':
                changed['patient_ids'] = changed['patient_ids'][::-1]
            else:
                changed['probabilities'][0, 0, 0] = np.nan if mutation == 'nan' else 2
            self.change_artifact(run, name, **changed)
            with self.assertRaisesRegex(ValueError, 'MC'):
                self.load()

    def test_quality_alignment_and_nonfinite_values_rejected(self):
        run = RUNS[0]; name = 'quality-features.npz'; original = load_npz(self.path.parent/run/name)
        for mutation in ('order', 'nan'):
            changed = copy.deepcopy(original)
            if mutation == 'order':
                changed['train_patient_id'] = changed['train_patient_id'][::-1]
            else:
                changed['train'][0, 0] = np.nan
            self.change_artifact(run, name, **changed)
            with self.assertRaisesRegex(ValueError, 'Quality|quality'):
                self.load()

    def test_independent_quality_and_prediction_row_orders_are_supported(self):
        run = RUNS[0]
        # Both V14 arrays retain their producer's row ordering; these need not
        # be the same. Reverse metadata and quality, leave sorted predictions.
        self.change_rows('validation', lambda rows: rows.reverse())
        for arm in RUNS:
            name = 'quality-features.npz'; quality = load_npz(self.path.parent/arm/name)
            for key in ('validation', 'validation_path', 'validation_patient_id',
                        'validation_eye', 'validation_component_id'):
                quality[key] = quality[key][::-1]
            self.change_artifact(arm, name, **quality)
        bundle = self.load()
        self.assertEqual(len(bundle[run]['data']['validation']['labels']), 60)

    def test_second_arm_sparse_support_blocks_before_first_fit(self):
        bundle = self.load()
        bundle[RUNS[1]]['data']['calibration']['labels'][:, 5] = 0
        with patch('trustretina_v15.v15.load_bundle', return_value=bundle), \
                patch('trustretina_v15.v15.fit_shared_temperature') as fit:
            with self.assertRaisesRegex(ValueError, 'both outcomes'):
                analyze(self.path, self.digest, PROTOCOL, self.approval, fixture=True)
            fit.assert_not_called()

    def test_both_candidates_fully_verified_before_any_fit(self):
        path = self.path.parent/RUNS[1]/'quality-features.npz'; path.unlink()
        with patch('trustretina_v15.v15.fit_shared_temperature') as fit:
            with self.assertRaises(ValueError):
                analyze(self.path, self.digest, PROTOCOL, self.approval, fixture=True)
            fit.assert_not_called()

    def test_sparse_support_and_unauthorized_fit(self):
        p = np.full((60, 8), .6); y = np.zeros((60, 8), dtype=int); groups = [f'g{i}' for i in range(60)]
        for role, allowed in (('validation', True), ('calibration', False), ('final_evaluation', True)):
            with self.assertRaises(ValueError):
                fit_shared_temperature(p, y, groups, role=role, allow_fitting=allowed)
        with self.assertRaisesRegex(ValueError, 'all outcomes'):
            fit_shared_temperature(p, y, groups, role='calibration', allow_fitting=True)
        y[::2] = 1
        with self.assertRaisesRegex(ValueError, '50 components'):
            fit_shared_temperature(p, y, ['one']*60, role='calibration', allow_fitting=True)

    def test_boundary_and_optimizer_failure_are_blockers(self):
        from types import SimpleNamespace
        y = np.tile(np.arange(60) % 2, (8, 1)).T; p = np.where(y, .8, .2)
        groups = [f'g{i}' for i in range(60)]
        for result in (SimpleNamespace(x=np.log(.05), fun=0., success=True, nfev=5),
                       SimpleNamespace(x=0., fun=1., success=False, nfev=5)):
            with patch('trustretina.reliability.development.minimize_scalar', return_value=result):
                with self.assertRaisesRegex(ValueError, 'optimizer failed'):
                    fit_shared_temperature(p, y, groups, role='calibration', allow_fitting=True)

    def test_metrics_have_exact_ten_bins_empty_bins_and_endpoint_handling(self):
        y = np.zeros((4, 8), dtype=int); p = np.tile([0., .1, .9, 1.], (8, 1)).T
        d = diagnostic(y, p, ['g1', 'g2', 'g3', 'g4'])
        row = d['per_class'][0]
        self.assertEqual([b['n'] for b in row['bins']], [1, 1, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertIsNone(row['bins'][2]['mean_probability'])
        self.assertAlmostEqual(row['ece'], .5)
        self.assertAlmostEqual(d['brier'], .455)
        self.assertTrue(row['sparse_or_single_outcome'])
        self.assertTrue(np.isfinite(d['patient_nll']))

    def test_zero_coverage_ties_missing_outcomes_and_uncertainty(self):
        y = np.zeros((4, 8), dtype=int); p = np.full((4, 8), .6)
        d = descriptive(y, p, np.tile(p, (20, 1, 1)), np.ones(4), ['g1', 'g2', 'g3', 'g4'])
        curve = d['curves']['confidence']
        self.assertEqual(curve['accepted'].tolist(), [4])
        self.assertIsNone(curve['fixed_coverage'][0]['hamming_risk'])
        h = d['classification']['per_class'][5]
        self.assertEqual(h['fp'], 4); self.assertEqual(h['fn'], 0)
        self.assertFalse(h['recall_defined']); self.assertIsNone(h['false_negative_rate'])
        np.testing.assert_allclose(d['mc_population_variance'], 0, atol=1e-30)

    def test_mean_eye_then_temperature_and_fixed_half_threshold(self):
        eyes = np.asarray([[.95]*8, [.4]*8]); t = 2.
        after_mean = temperature_probabilities(eyes.mean(0)[None], t)
        mean_after = temperature_probabilities(eyes, t).mean(0)
        self.assertFalse(np.allclose(after_mean, mean_after))
        p = np.asarray([[0., .499, .5, .501, 1., .1, .9, .4]])
        for t in (.05, 1., 20.):
            np.testing.assert_array_equal(p >= .5, temperature_probabilities(p, t) >= .5)

    def test_complete_fixture_separate_fits_retains_regressions_and_reproducibility(self):
        # V14 records source_commit in stage-report, not experiment-spec.
        self.assertNotIn('source_commit', json.loads((self.path.parent/RUNS[0]/'experiment-spec.json').read_text()))
        first = analyze(self.path, self.digest, PROTOCOL, self.approval, fixture=True)
        second = analyze(self.path, self.digest, PROTOCOL, self.approval, fixture=True)
        self.assertEqual(json.dumps(first, default=lambda v: v.tolist(), sort_keys=True),
                         json.dumps(second, default=lambda v: v.tolist(), sort_keys=True))
        self.assertEqual(first['data_kind'], FIXTURE)
        self.assertEqual(first['deployment_default'], 'REVIEW REQUIRED')
        self.assertFalse(first['threshold_fitted']); self.assertFalse(first['historical_final_accessed'])
        for run in RUNS:
            fit = first['runs'][run]['fit']
            self.assertTrue(.05 < fit['temperature'] < 20)
            self.assertIn('in-sample', fit['interpretation'])
            for role in ('validation', 'calibration'):
                self.assertTrue(first['runs'][run]['roles'][role]['all_binary_decisions_unchanged'])
        diff = first['paired_comparisons']['validation']['raw']['per_class']['H']
        self.assertTrue(diff['point_regression'])
        self.assertGreater(diff['fn'], 0)
        with patch('trustretina_v15.v15.fit_shared_temperature', wraps=fit_shared_temperature) as fit:
            analyze(self.path, self.digest, PROTOCOL, self.approval, fixture=True)
            self.assertEqual(fit.call_count, 2)
            self.assertTrue(all(call.kwargs['role'] == 'calibration' for call in fit.call_args_list))

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Optional report plotting dependency')
    def test_report_plots_hashes_and_overwrite_refusal(self):
        report = analyze(self.path, self.digest, PROTOCOL, self.approval, fixture=True)
        output = self.root/'report'; save_report(report, output)
        self.assertEqual(len(list(output.glob('*.png'))), 8)
        hashes = json.loads((output/'output-hashes.json').read_text())
        for name, digest in hashes.items():
            self.assertEqual(sha256(output/name), digest)
        with self.assertRaises(FileExistsError):
            save_report(report, output)

    def test_cli_readiness_has_no_real_execution_command(self):
        command = [sys.executable, str(ROOT/'scripts/calibration_reliability_v15.py')]
        result = subprocess.run(command+['readiness'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)['real_execution_ready'])
        result = subprocess.run(command+['fit-real'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

    def test_analysis_import_does_not_load_torch_or_runner_clients(self):
        code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torchvision', 'kaggle', 'kagglesdk'):
            raise ImportError('Disabled in numeric analysis')
sys.meta_path.insert(0, Block())
import trustretina_v15.v15
assert 'torch' not in sys.modules
assert 'kaggle' not in sys.modules
'''
        import os
        env = {**os.environ, 'PYTHONPATH': str(ROOT/'scripts')+os.pathsep+str(ROOT/'src')}
        result = subprocess.run([sys.executable, '-B', '-c', code], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
