"""CPU checks of the v14 report's multilabel counts and uncalibrated fixed threshold."""
import importlib.util,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('report_candidate_results',ROOT/'scripts/report_candidate_results.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class CandidateReportTests(unittest.TestCase):
    def test_exact_binary_counts_and_micro_formula(self):
        y=np.array([[1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1],[1,0,0,1,1,0,0,1]])
        p=np.array([[.9,.8,.8,.1,.8,.2,.2,.1],[.1,.8,.1,.8,.2,.8,.1,.9],[.2,.1,.2,.8,.9,.1,.8,.9]])
        metrics,rows=mod.summarize_validation(y,p)
        self.assertEqual(rows[0]['TP'],1);self.assertEqual(rows[0]['FP'],0);self.assertEqual(rows[0]['FN'],1);self.assertEqual(rows[0]['TN'],1)
        tp=sum(r['TP'] for r in rows);fp=sum(r['FP'] for r in rows);fn=sum(r['FN'] for r in rows)
        self.assertAlmostEqual(metrics['micro_f1'],2*tp/(2*tp+fp+fn))
        self.assertAlmostEqual(metrics['macro_f1'],np.mean([r['F1'] for r in rows]))
    def test_threshold_is_inclusive_half_without_fitting(self):
        y=np.array([[1]*8,[0]*8]);p=np.array([[.5]*8,[.49]*8]);metrics,rows=mod.summarize_validation(y,p)
        self.assertEqual(metrics['macro_f1'],1);self.assertTrue(all(r['TP']==1 and r['FP']==0 for r in rows))
    def test_mixed_label_containment_counts_are_not_voted(self):
        y=np.array([[1]*8,[0]*8,[1]*8]);p=np.array([[.8]*8,[.7]*8,[.2]*8])
        _,rows=mod.summarize_validation(y,p,['shared','shared','other'])
        self.assertEqual(rows[0]['positive_components'],2);self.assertEqual(rows[0]['negative_components'],1)
        self.assertEqual(rows[0]['FPR'],1);self.assertEqual(rows[0]['FNR'],.5)
    def test_rejects_non_multilabel_shape(self):
        with self.assertRaises(ValueError):mod.summarize_validation([1,0],[.8,.2])

if __name__=='__main__':unittest.main()
