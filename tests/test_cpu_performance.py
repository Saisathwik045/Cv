"""No real-model timing in CI: test bounds, identity refusal and output safety."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from dashboard.performance import summarize, measure
from trustretina.inference.bundle import verified_contract

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('benchmark_demo_cpu', ROOT/'scripts/benchmark_demo_cpu.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class PerformanceContracts(unittest.TestCase):
    def test_public_observation_matches_its_exact_harness_and_documented_values(self):
        raw=(ROOT/'results/demo-cpu-performance/measurement.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), '5830473786b52d697935927db9c39966ac091d16688a9e7074d4980550d6891f')
        data=json.loads(raw);doc=(ROOT/'docs/CPU_PERFORMANCE.md').read_text()
        self.assertFalse(data['clinical_validation']);self.assertFalse(data['research_evaluation'])
        self.assertEqual(data['measurements']['calls'],{'model_loads':1,'predictions':4,'paired_explanations':1})
        for key, path in (('harness_sha256','scripts/benchmark_demo_cpu.py'),
                          ('measurement_module_sha256','dashboard/performance.py')):
            self.assertEqual(data['provenance'][key],hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
        for values in data['measurements'].values():
            if isinstance(values,dict) and 'median_seconds' in values:
                self.assertIn(f"{values['median_seconds']:.6f}",doc)
        for forbidden in ('/workspace/', '/home/', 'raw_probabilities', 'calibrated_probabilities'):
            self.assertNotIn(forbidden,raw.decode())

    def test_statistics_do_not_invent_tail_percentiles(self):
        self.assertEqual(summarize([3., 1., 2.]), {'samples_seconds':[3.,1.,2.], 'count':3,
                         'minimum_seconds':1., 'median_seconds':2., 'maximum_seconds':3.})
        for invalid in ([], [-1], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                summarize(invalid)

    def test_output_is_fresh_private_outside_checkout_and_not_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); p = root/'result.json'
            self.assertEqual(cli.private_output(p), p)
            p.write_text('existing')
            with self.assertRaises(ValueError): cli.private_output(p)
            self.assertEqual(p.read_text(), 'existing')
            link=root/'linked';link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):cli.private_output(link/'new.json')
            public=root/'public';public.mkdir(mode=0o755);public.chmod(0o755)
            with self.assertRaises(ValueError):cli.private_output(public/'new.json')
        with self.assertRaises(ValueError):cli.private_output(ROOT/'timing.json')
        with self.assertRaises(ValueError):cli.private_output('relative.json')

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'optional image dependency')
    def test_fixed_workload_discards_predictions_and_checks_unchanged_state(self):
        contract=verified_contract()
        result={'device':'cpu','model':'convnext_tiny','best_epoch':2,'class_order':list('NDGCAHMO'),
                'mc_passes':20,'threshold':.5,'decision':'REVIEW REQUIRED',
                **{k:contract[k] for k in ('temperature','checkpoint_sha256','preprocessing_sha256')},
                'raw_probabilities':dict.fromkeys('NDGCAHMO',.4),
                'calibrated_probabilities':dict.fromkeys('NDGCAHMO',.3)}
        service=Mock(state_sha256='frozen');service.predict.return_value=result
        service.explain.return_value=({},[b'left',b'right'])
        service.model.modules.return_value=[]
        factory=Mock(return_value=service)
        clock=Mock(side_effect=range(12))
        report=measure(factory,contract,lambda model:'frozen',clock=clock)
        factory.assert_called_once();self.assertEqual(service.predict.call_count,4)
        service.explain.assert_called_once()
        self.assertEqual(report['calls'],{'model_loads':1,'predictions':4,'paired_explanations':1})
        self.assertEqual(report['repeated_prediction_seconds']['count'],3)
        self.assertNotIn('raw_probabilities',str(report))
        service.reset_mock();service.predict.return_value={**result,'device':'cuda'}
        with self.assertRaises(ValueError):measure(lambda:service,contract,lambda m:'frozen')
        service.explain.assert_not_called()
        service.predict.return_value=result
        with self.assertRaisesRegex(ValueError,'state changed'):
            measure(lambda:service,contract,lambda m:'changed')


if __name__ == '__main__':
    unittest.main()
