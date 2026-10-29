import copy, tempfile, unittest
from pathlib import Path
from trustretina.config import CLASSES, PARTITIONS
from trustretina.data.development import (
    development_assignment,
    audit_development,
    write_diagnostic,
)
from trustretina.data.validation import DataValidationError


class DevelopmentProtocolTests(unittest.TestCase):
    def fixture(self):
        patients = []
        for i in range(160):
            y = [0] * 8
            y[i % 8] = 1
            patients.append({"patient_id": str(i), "labels": y})
        inv = {"patients": patients, "fingerprint": "synthetic"}
        config = {
            "seed": 2026,
            "split": {
                "fractions": [0.4, 0.1, 0.4, 0.1],
                "minimum_positive_patients": 1,
                "minimum_negative_patients": 1,
            },
        }
        return inv, config

    def test_deterministic_grouped_all_eight_support(self):
        inv, c = self.fixture()
        link = {
            "a": {"patient_id": "0"},
            "b": {"patient_id": "8"},
            "conflicting_patient_labels": False,
        }
        a = development_assignment(inv, [link], c)
        b = development_assignment(inv, [link], c)
        self.assertEqual(a, b)
        owner = {p: r for r, ids in a["partitions"].items() for p in ids}
        self.assertEqual(owner["0"], owner["8"])
        result = audit_development(
            a,
            inv,
            [link],
            [],
            review_complete=True,
            controls_passed=True,
            raw_verified=True,
        )
        self.assertTrue(result["pilot_ready"])
        self.assertFalse(result["full_training_ready"])
        for s in result["support"].values():
            self.assertGreaterEqual(min(s["positive"]), 1)
            self.assertGreaterEqual(min(s["negative"]), 1)

    def test_specific_unresolved_crossing_blocks_without_mass_quarantine(self):
        inv, c = self.fixture()
        m = development_assignment(inv, [], c)
        roles = list(m["partitions"])
        a = m["partitions"][roles[0]][0]
        b = m["partitions"][roles[1]][0]
        pending = {"a": {"patient_id": a}, "b": {"patient_id": b}, "review_index": 42}
        result = audit_development(
            m,
            inv,
            [],
            [pending],
            review_complete=True,
            controls_passed=True,
            raw_verified=True,
        )
        self.assertFalse(result["pilot_ready"])
        self.assertEqual(result["crossing_cases"], [pending])
        self.assertEqual(sum(len(ids) for ids in m["partitions"].values()), 160)
        with tempfile.TemporaryDirectory() as d:
            out = write_diagnostic(Path(d) / "v3", m, result)
            self.assertTrue((out / "readiness.sha256").is_file())
            self.assertFalse((out / "complete.json").exists())
            with self.assertRaises(FileExistsError):
                write_diagnostic(out, m, result)

    def test_tampered_support_or_group_is_rejected(self):
        inv, c = self.fixture()
        m = development_assignment(inv, [], c)
        m["support"]["train"]["positive"][0] += 1
        with self.assertRaises(DataValidationError):
            audit_development(
                m,
                inv,
                [],
                [],
                review_complete=True,
                controls_passed=True,
                raw_verified=True,
            )

    def test_insufficient_calibration_or_failed_control_not_overridden(self):
        inv, c = self.fixture()
        c["split"]["fractions"] = [0.65, 0.15, 0.1, 0.1]
        m = development_assignment(inv, [], c)
        result = audit_development(
            m,
            inv,
            [],
            [],
            review_complete=True,
            controls_passed=False,
            raw_verified=True,
        )
        self.assertIn("photographic_controls_failed", result["blockers"])
        self.assertIn("insufficient_groups_for_shared_calibration", result["blockers"])

    def test_real_data_blocker_prevents_any_frozen_output(self):
        from trustretina.data.development import freeze_for_pilot

        inv, c = self.fixture()
        m = development_assignment(inv, [], c)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "frozen"
            with self.assertRaises(DataValidationError):
                freeze_for_pilot(
                    out,
                    m,
                    {"checks_passed": False},
                    inv,
                    config_sha256="a" * 64,
                    evidence_sha256="b" * 64,
                )
            self.assertFalse(out.exists())

    def test_configuration_has_no_per_class_calibration_or_final_selection(self):
        from trustretina.data.development import load_development_config

        path = Path(__file__).resolve().parents[1] / "configs/development-v3.toml"
        c = load_development_config(path)
        self.assertFalse(c["calibration"]["class_specific_fitting"])
        self.assertFalse(c["pilot"]["uses_final_evaluation"])
        self.assertEqual(c["pilot"]["maximum_epochs"], 2)

    def test_serialized_manifest_roundtrip_reaudits_and_conflicts_stay_explicit(self):
        import json

        inv, c = self.fixture()
        links = [
            {
                "a": {"patient_id": "0"},
                "b": {"patient_id": "1"},
                "conflicting_patient_labels": True,
            }
        ]
        m = development_assignment(inv, links, c)
        serialized = json.loads(json.dumps(m))
        result = audit_development(
            serialized,
            inv,
            links,
            [],
            review_complete=True,
            controls_passed=True,
            raw_verified=True,
        )
        self.assertEqual(result["verified_link_annotation_conflicts"], 1)
        self.assertFalse(result["pilot_ready"])
        self.assertEqual(inv["patients"][0]["labels"], [1, 0, 0, 0, 0, 0, 0, 0])
