"""Research-critical configuration, planning and capability failure paths."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from trustretina.cli import main
from trustretina.config import BACKBONES, CLASSES, load_config
from trustretina.environment import require_gpu
from trustretina.planning import execution_plan

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase0.toml"


class FoundationTests(unittest.TestCase):
    def test_fixed_protocol_and_portable_paths(self):
        config = load_config(CONFIG)
        self.assertEqual(config['class_order'], CLASSES)
        self.assertEqual(config['experiment']['backbones'], BACKBONES)
        self.assertEqual(Path(config['data']['root']), ROOT / 'data/raw/ODIR-5K')
        self.assertEqual(Path(config['paths']['artifacts']), ROOT / 'artifacts')
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, '-m', 'trustretina', 'check-config',
                                     '--config', str(CONFIG)], cwd=folder, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['data'], config['data'])

    def reject_mutation(self, old, new):
        with tempfile.TemporaryDirectory() as folder:
            candidate = Path(folder) / 'bad.toml'
            original = CONFIG.read_text()
            self.assertIn(old, original)
            candidate.write_text(original.replace(old, new))
            with self.assertRaises((ValueError, KeyError, TypeError)):
                load_config(candidate)

    def test_reject_class_reordering(self):
        self.reject_mutation('["N", "D", "G", "C", "A", "H", "M", "O"]',
                             '["D", "N", "G", "C", "A", "H", "M", "O"]')

    def test_require_four_disjoint_partition_roles(self):
        self.reject_mutation('"calibration", "final_evaluation"', '"validation", "test"')

    def test_reject_bad_fractions_and_support(self):
        for old, new in [('0.65, 0.15, 0.10, 0.10', '0.65, 0.15, 0.10, 0.20'),
                         ('0.65, 0.15, 0.10, 0.10', '0.75, 0.15, 0.10, 0.00'),
                         ('minimum_positive_patients = 1', 'minimum_positive_patients = 0')]:
            with self.subTest(new=new):
                self.reject_mutation(old, new)

    def test_reject_non_authoritative_labels_and_final_selection(self):
        for old, new in [('labels = "data.xlsx"', 'labels = "processed.csv"'),
                         ('label_source = "patient"', 'label_source = "eye"'),
                         ('validation_macro_f1', 'final_evaluation_macro_f1')]:
            with self.subTest(new=new):
                self.reject_mutation(old, new)

    def test_reject_invalid_training_and_stochastic_settings(self):
        for old, new in [('batch_size = 16', 'batch_size = 0'),
                         ('learning_rate = 0.0001', 'learning_rate = nan'),
                         ('mc_dropout_passes = 30', 'mc_dropout_passes = 1'),
                         ('dropout_probability = 0.2', 'dropout_probability = 1.0')]:
            with self.subTest(new=new):
                self.reject_mutation(old, new)

    def test_plan_is_deterministic_without_execution_claims(self):
        config = load_config(CONFIG)
        plan = execution_plan(config)
        self.assertEqual(plan, execution_plan(config))
        self.assertEqual(len(plan['jobs']), 5)
        self.assertEqual(len({job['run_key'] for job in plan['jobs']}), 5)
        self.assertTrue(all(job['status'] == 'not_implemented' for job in plan['jobs']))
        changed = copy.deepcopy(config)
        changed['seed'] += 1
        self.assertNotEqual(plan['config_sha256'], execution_plan(changed)['config_sha256'])
        self.assertEqual(plan['kind'], 'declarative_plan_only')

    def test_missing_config_exits_nonzero(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['check-config', '--config', '/nonexistent/trustretina.toml']), 2)

    def test_gpu_gate_does_not_infer_access(self):
        self.assertFalse(require_gpu({'cuda_available': None, 'gpu_devices': []}))
        self.assertFalse(require_gpu({'cuda_available': False, 'gpu_devices': []}))
        self.assertFalse(require_gpu({'cuda_available': True, 'gpu_devices': []}))
        with patch('trustretina.cli.environment_report', return_value={'cuda_available': None}):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['check-env', '--require-gpu']), 1)

    def test_unimplemented_training_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                main(['train'])
        self.assertEqual(result.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
