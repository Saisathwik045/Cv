"""Scientific chart semantics: reference/prediction orientation and integer counts."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('altair') and importlib.util.find_spec('pandas'),
                     'optional dashboard chart dependencies')
class ChartContracts(unittest.TestCase):
    def test_binary_matrix_orientation_and_count_conservation(self):
        from dashboard.charts import confusion_rows
        rows = confusion_rows({'positive': 10, 'negative': 10,
                               'false_negative': 2, 'predicted_positive': 11})
        self.assertEqual({(r['Actual'], r['Predicted']): (r['Outcome'], r['Patients']) for r in rows},
                         {('Absent', 'Absent'): ('TN', 7), ('Absent', 'Present'): ('FP', 3),
                          ('Present', 'Absent'): ('FN', 2), ('Present', 'Present'): ('TP', 8)})
        self.assertEqual(sum(r['Patients'] for r in rows), 20)

    def test_weighted_or_impossible_counts_are_not_presented_as_patients(self):
        from dashboard.charts import confusion_rows
        for counts in ({'positive': 10, 'negative': 10, 'false_negative': 11, 'predicted_positive': 5},
                       {'positive': 10, 'negative': 10, 'false_negative': 2, 'predicted_positive': 21},
                       {'positive': 10.5, 'negative': 10, 'false_negative': 2, 'predicted_positive': 11}):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                confusion_rows(counts)

    def test_probability_comparison_keeps_class_order_and_independent_scores(self):
        from dashboard.charts import probability_chart
        labels = list('NDGCAHMO')
        spec = probability_chart({'class_order': labels,
                                  'raw_probabilities': dict.fromkeys(labels, .7),
                                  'calibrated_probabilities': dict.fromkeys(labels, .8),
                                  'threshold': .5}).to_dict()
        bars = spec['layer'][0]
        self.assertEqual(bars['encoding']['y']['sort'], labels)
        self.assertEqual(bars['encoding']['x']['scale']['domain'], [0, 1])
        rows = spec['datasets'][bars['data']['name']]
        self.assertEqual(len(rows), 16)
        self.assertEqual([r['Probability'] for r in rows if r['Series'] == 'Calibrated'], [.8] * 8)


if __name__ == '__main__':
    unittest.main()
