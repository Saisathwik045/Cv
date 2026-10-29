import unittest, copy
from trustretina.data.contained_release import (
    merge_scheduled,
    group_constraints,
    audit_and_assign,
)
from trustretina.data.validation import DataValidationError
from trustretina.data.bounded_review import POLICY_V3


class ContainedReleaseTests(unittest.TestCase):
    def fixture(self):
        catalog = [
            {"patient_id": str(i), "eye": "left", "sha256": str(i)} for i in range(160)
        ]
        schedule = [
            {
                "review_index": i,
                "a_index": i % 159,
                "b_index": i % 159 + 1,
                "pair_index": i,
                "candidate": True,
            }
            for i in range(1536)
        ]
        selection = {"schedule": schedule, "completed_indices": list(range(422))}
        rows = [
            dict(
                r,
                a_photo={"raw_sha256": catalog[r["a_index"]]["sha256"]},
                b_photo={"raw_sha256": catalog[r["b_index"]]["sha256"]},
                coarse_intensity_correlation=0.5,
                coarse_gradient_correlation=0.2,
                independent_evidence={
                    "link": False,
                    "policy_version": POLICY_V3["version"],
                    "reason": "synthetic unresolved",
                },
            )
            for r in schedule
        ]
        return catalog, selection, rows

    def test_exact_schedule_completion_and_missing_resume(self):
        catalog, selection, rows = self.fixture()
        merged, missing = merge_scheduled(selection, catalog, [rows[:422], rows[422:]])
        self.assertEqual(len(merged), 1536)
        self.assertFalse(missing)
        _, missing = merge_scheduled(selection, catalog, [rows[:422]])
        self.assertEqual(len(missing), 1114)
        with self.assertRaises(DataValidationError):
            merge_scheduled(selection, catalog, [rows, rows[:1]])

    def test_no_candidate_expansion_or_identity_mutation(self):
        c, s, rows = self.fixture()
        rows[0]["review_index"] = 99999
        with self.assertRaises(DataValidationError):
            merge_scheduled(s, c, [rows])
        c, s, rows = self.fixture()
        rows[0]["a_photo"]["raw_sha256"] = "changed"
        with self.assertRaises(DataValidationError):
            merge_scheduled(s, c, [rows])

    def test_uncertainty_containment_is_not_confirmed_duplication(self):
        c, s, rows = self.fixture()
        patients = [
            {"patient_id": str(i), "labels": [int(j == i % 8) for j in range(8)]}
            for i in range(160)
        ]
        inv = {"patients": patients, "fingerprint": "synthetic"}
        edges, decisions = group_constraints(
            inv, c, rows[2:3], named_cases=[{"review_index": 2}]
        )
        self.assertEqual(edges[0]["relationship"], "unresolved_content_containment")
        self.assertTrue(edges[0]["conflicting_patient_labels"])
        self.assertEqual(decisions[0]["decision"], "contain_unresolved")
        config = {
            "seed": 2026,
            "split": {
                "fractions": [0.4, 0.1, 0.4, 0.1],
                "minimum_positive_patients": 1,
                "minimum_negative_patients": 1,
            },
        }
        m, a = audit_and_assign(
            inv,
            edges,
            rows,
            [],
            config,
            validation_reused=True,
            budget_seconds=4500,
            receipts_verified=True,
            named_review_complete=True,
        )
        self.assertTrue(a["checks_passed"])
        self.assertFalse(a["annotation_errors_inferred"])
        self.assertFalse(a["all_possible_duplicates_ruled_out"])
        owner = {p: r for r, ids in m["partitions"].items() for p in ids}
        self.assertEqual(owner["2"], owner["3"])

    def test_missing_coverage_and_budget_are_real_blockers(self):
        c, s, rows = self.fixture()
        inv = {
            "patients": [
                {"patient_id": str(i), "labels": [int(j == i % 8) for j in range(8)]}
                for i in range(160)
            ],
            "fingerprint": "synthetic",
        }
        config = {
            "seed": 2026,
            "split": {
                "fractions": [0.4, 0.1, 0.4, 0.1],
                "minimum_positive_patients": 1,
                "minimum_negative_patients": 1,
            },
        }
        _, audit = audit_and_assign(
            inv,
            [],
            rows[:422],
            list(range(422, 1536)),
            config,
            validation_reused=True,
            budget_seconds=7201,
            receipts_verified=True,
            named_review_complete=False,
        )
        self.assertFalse(audit["pilot_ready"])
        self.assertIn("scheduled_review_incomplete", audit["blockers"])
        self.assertIn("additional_cpu_budget_exceeded", audit["blockers"])
