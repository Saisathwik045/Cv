"""Synthetic mutation tests for research-only plan and artifact-inventory gates."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('check_candidate_plan',ROOT/'scripts/check_candidate_plan.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class CandidatePlanTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((ROOT/'results/candidate-v14/proposed-plan.json').read_text())
        self.datasets=json.loads((ROOT/'results/candidate-v14/dataset-options.json').read_text())

    def test_valid_proposal_does_not_imply_execution_ready(self):
        result=module.validate(self.plan,self.datasets)
        self.assertFalse(result['execution_ready']);self.assertEqual(result['GPU_jobs_authorized'],0)

    def test_fitting_authorization_cannot_be_inferred_from_budget(self):
        self.plan['authorization']['submit_gpu_jobs']=True
        with self.assertRaisesRegex(ValueError,'authorization'):module.validate(self.plan,self.datasets)

    def test_calibration_or_final_cannot_select_checkpoint(self):
        for role in ('calibration','historical_final'):
            p=copy.deepcopy(self.plan)
            for run in p['runs']:run['settings']['selection_role']=role
            with self.assertRaisesRegex(ValueError,'role leakage'):module.validate(p,self.datasets)

    def test_unknown_labels_cannot_be_converted_to_negative(self):
        self.plan['evaluation']['unknown_labels']='negative'
        with self.assertRaisesRegex(ValueError,'Unknown labels'):module.validate(self.plan,self.datasets)

    def test_unverified_dataset_cannot_be_called_full_validation(self):
        self.datasets['options'][0]['full_eight_compatible_verified']=True
        with self.assertRaisesRegex(ValueError,'invent'):module.validate(self.plan,self.datasets)

    def test_loss_cannot_change_under_the_same_proposal(self):
        for run in self.plan['runs']:run['settings']['loss']='sqrt_ratio_bce'
        with self.assertRaisesRegex(ValueError,'settings changed'):module.validate(self.plan,self.datasets)

    def test_resume_budget_includes_overhead(self):
        self.plan['compute']['optional_single_resume_max_worker_seconds']=3000
        with self.assertRaisesRegex(ValueError,'cap'):module.validate(self.plan,self.datasets)


if __name__=='__main__':unittest.main()
