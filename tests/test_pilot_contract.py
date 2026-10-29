import unittest
from unittest.mock import patch
from trustretina.runner.pilot_runtime import validate_pilot_spec, run_pilot


class PilotContractTests(unittest.TestCase):
    def spec(self):
        rows = lambda p: [
            dict(
                patient_id=p,
                eye=eye,
                labels=[1, 0, 0, 0, 0, 0, 0, 0],
                path=p + "_" + eye + ".jpg",
                component_id=p,
            )
            for eye in ("left", "right")
        ]
        return dict(
            scope="one_efficientnet_b0_execution_pilot_v4",
            data_checks_passed=True,
            final_evaluation_access=False,
            epochs=2,
            backbone="efficientnet_b0",
            initialization="random_unfitted_execution_only",
            batch_size=16,
            split_sha256="a" * 64,
            readiness_sha256="b" * 64,
            train_rows=rows("synthetic-train"),
            validation_rows=rows("synthetic-validation"),
        )

    def test_guard_before_cuda_or_training(self):
        spec = self.spec()
        spec["data_checks_passed"] = False
        with self.assertRaises(PermissionError):
            run_pilot(
                spec,
                input_root="/nonexistent",
                output="/nonexistent",
                nonce="synthetic",
                source_commit="synthetic",
            )

    def test_no_final_access_or_extra_epochs(self):
        spec = self.spec()
        self.assertEqual(len(validate_pilot_spec(spec)), 4)
        for field, value in [
            ("epochs", 3),
            ("final_evaluation_access", True),
            ("final_rows", []),
        ]:
            changed = dict(spec)
            changed[field] = value
            with self.assertRaises(PermissionError):
                validate_pilot_spec(changed)

    def test_group_leakage_and_missing_eye_rejected(self):
        spec = self.spec()
        spec["validation_rows"][0]["component_id"] = spec["train_rows"][0][
            "component_id"
        ]
        with self.assertRaises(PermissionError):
            validate_pilot_spec(spec)
        spec = self.spec()
        spec["train_rows"].pop()
        with self.assertRaises(PermissionError):
            validate_pilot_spec(spec)
