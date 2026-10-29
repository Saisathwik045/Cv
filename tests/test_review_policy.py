import unittest
from trustretina.data.review_policy import (
    review_candidates,
    assert_partition_separation,
)


class ReviewPolicyTests(unittest.TestCase):
    def fixture(self):
        patients = [{"patient_id": str(i), "labels": [i % 2] * 8} for i in range(4)]

        def pair(a, b, exact=False):
            return {
                "a": {"patient_id": str(a), "eye": "left", "sha256": str(a)},
                "b": {"patient_id": str(b), "eye": "right", "sha256": str(b)},
                "file_equal": exact,
                "pixel_equal": False,
                "conflicting_patient_labels": a % 2 != b % 2,
            }

        return patients, [pair(0, 1, True), pair(1, 2), pair(2, 3)]

    def test_transitive_verified_group_does_not_promote_weak_hash(self):
        patients, pairs = self.fixture()
        decision = dict(
            pairs[1],
            pair_index=1,
            independent_evidence={
                "classification": "strongly_supported_probable_copy",
                "reason": "controlled feature/photometric evidence",
                "controls_passed": True,
            },
        )
        result = review_candidates(patients, pairs, [decision])
        self.assertIn(["0", "1", "2"], result["must_link_patient_groups"])
        self.assertEqual(result["quarantine_patients"], ["0", "1", "2", "3"])
        self.assertFalse(result["training_ready"])
        with self.assertRaises(ValueError):
            assert_partition_separation(
                {"train": ["0", "1"], "validation": ["2"]},
                result["must_link_patient_groups"],
            )

    def test_false_positive_removal_and_unreviewed_quarantine(self):
        patients, pairs = self.fixture()
        decision = dict(
            pairs[1],
            pair_index=1,
            independent_evidence={
                "classification": "evidence_supported_false_positive",
                "reason": "reviewed different content plus independent evidence",
                "controls_passed": True,
            },
        )
        result = review_candidates(patients, pairs, [decision])
        self.assertEqual(result["eligible_patients"], ["0", "1"])
        self.assertEqual(result["quarantine_patients"], ["2", "3"])
        self.assertTrue(
            assert_partition_separation(
                {"train": ["0", "1"]},
                result["must_link_patient_groups"],
                result["quarantine_patients"],
            )
        )
        with self.assertRaises(ValueError):
            assert_partition_separation(
                {"train": ["0", "1", "2"]},
                result["must_link_patient_groups"],
                result["quarantine_patients"],
            )

    def test_identity_binding_and_no_exact_promotion(self):
        patients, pairs = self.fixture()
        decision = dict(
            pairs[1],
            pair_index=1,
            independent_evidence={
                "classification": "confirmed_duplicate",
                "reason": "hash alone",
                "controls_passed": True,
            },
        )
        with self.assertRaises(ValueError):
            review_candidates(patients, pairs, [decision])
        decision["a"] = dict(decision["a"], sha256="tampered")
        with self.assertRaises(ValueError):
            review_candidates(patients, pairs, [decision])
