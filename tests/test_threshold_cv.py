import unittest
import numpy as np
from trustretina.evaluation.threshold_cv import (
    evaluate_folds, fit_thresholds, make_group_folds, summarize, validate_folds,
)


class ThresholdCVTests(unittest.TestCase):
    def test_exact_f1_tie_chooses_nearest_half_then_higher(self):
        # .45 predicts 20 TP / 20 FP; .55 predicts 10 TP / 0 FP / 10 FN.
        # Both are exactly F1=2/3; .50 predicts 10 TP / 20 FP and is worse.
        column = np.tile([1, 1, 0, 0], 10)
        y = np.repeat(column[:, None], 8, axis=1)
        p = np.repeat(np.tile([.46, .8, .52, .52], 10)[:, None], 8, axis=1)
        thresholds, details = fit_thresholds(y, p, [str(i) for i in range(40)])
        np.testing.assert_array_equal(thresholds, [.55] * 8)
        self.assertFalse(details[0]['fallback'])
        self.assertEqual(details[0]['positive_components'], 20)
        # All grid values through .90 give perfect predictions: choose .50.
        p = np.where(y == 1, .9, 0.)
        thresholds, _ = fit_thresholds(y, p, [str(i) for i in range(40)])
        np.testing.assert_array_equal(thresholds, [.5] * 8)

    def test_sparse_group_support_falls_back_despite_many_positive_patients(self):
        y = np.repeat(np.array([1] * 20 + [0] * 20)[:, None], 8, axis=1)
        p = np.where(y == 1, .2, .1)
        groups = ['one-positive-group'] * 20 + [str(i) for i in range(20)]
        t, details = fit_thresholds(y, p, groups)
        np.testing.assert_array_equal(t, [.5] * 8)
        self.assertTrue(details[5]['fallback'])
        self.assertEqual(details[5]['positive'], 20)
        self.assertEqual(details[5]['positive_components'], 1)

    def test_allocator_keeps_mixed_label_components_and_is_order_invariant(self):
        y = np.repeat(np.tile([0, 1], 20)[:, None], 8, axis=1)
        patients = np.array([f'p{i:02}' for i in range(40)])
        groups = np.array([f'g{i // 2:02}' for i in range(40)])
        folds, _ = make_group_folds(y, patients, groups)
        for group in set(groups):
            self.assertEqual(len(set(folds[groups == group])), 1)
        reverse = np.arange(39, -1, -1)
        reversed_folds, _ = make_group_folds(y[reverse], patients[reverse], groups[reverse])
        np.testing.assert_array_equal(folds[reverse], reversed_folds)

    def test_assessment_scores_and_labels_cannot_change_own_fitted_threshold(self):
        column = np.tile([1] * 10 + [0] * 10, 4)
        y = np.repeat(column[:, None], 8, axis=1)
        p = np.where(y == 1, .4, .2)
        ids = np.array([str(i) for i in range(80)])
        folds = np.repeat(np.arange(4), 20)
        report, decisions, _ = evaluate_folds(y, p, ids, ids, folds, role='validation', allow_exploratory=True)
        self.assertEqual(report['threshold_cv']['macro_f1'], 1.)
        self.assertEqual(report['baseline']['macro_f1'], 0.)
        np.testing.assert_array_equal(decisions, y)
        changed_p = p.copy()
        changed_p[folds == 0] = np.where(y[folds == 0] == 1, .9, .7)
        changed_y = y.copy()
        changed_y[folds == 0] = 1 - y[folds == 0]
        altered, _, _ = evaluate_folds(changed_y, changed_p, ids, ids, folds, role='validation', allow_exploratory=True)
        self.assertEqual(report['folds'][0]['fitting'], altered['folds'][0]['fitting'])
        self.assertNotEqual(report['folds'][0]['threshold_cv'], altered['folds'][0]['threshold_cv'])
        self.assertTrue(all(f['group_overlap'] == 0 for f in report['folds']))

    def test_group_leakage_and_duplicate_patients_rejected(self):
        y = np.repeat(np.tile([0, 1], 8)[:, None], 8, axis=1)
        ids = np.array([str(i) for i in range(16)])
        folds = np.repeat(np.arange(4), 4)
        groups = ids.copy()
        groups[4] = groups[0]
        with self.assertRaisesRegex(ValueError, 'crosses'):
            validate_folds(y, ids, groups, folds)
        ids[1] = ids[0]
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            make_group_folds(y, ids, groups)

    def test_exact_multilabel_counts_and_permission_boundary(self):
        y = np.zeros((3, 8), int)
        y[:, 0] = [1, 1, 0]
        pred = np.zeros((3, 8), int)
        pred[:, 0] = [1, 0, 1]
        summary = summarize(y, pred)
        row = summary['per_class'][0]
        self.assertEqual((row['tp'], row['fp'], row['fn']), (1, 1, 1))
        self.assertEqual(summary['micro_f1'], .5)
        self.assertEqual(summary['macro_f1'], .5 / 8)
        self.assertEqual(summary['subset_exact_match_accuracy'], 1 / 3)
        with self.assertRaises(PermissionError):
            evaluate_folds(y, pred, ['a', 'b', 'c'], ['a', 'b', 'c'], [0, 1, 2], role='final_evaluation', allow_exploratory=True)
        with self.assertRaises(ValueError):
            fit_thresholds(y, np.full_like(y, np.nan, dtype=float), ['a', 'b', 'c'])
