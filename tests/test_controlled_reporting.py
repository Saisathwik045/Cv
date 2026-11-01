import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from report_controlled_results import confusion_rows


class ControlledReportingTests(unittest.TestCase):
    def test_exact_multilabel_confusion_and_fixed_point_five(self):
        y=np.zeros((3,8),int);y[:,0]=[1,1,0];y[:,1]=[0,1,0]
        p=np.zeros((3,8));p[:,0]=[.8,.2,.5];p[:,1]=[.2,.9,.4]
        rows=confusion_rows(y,p);n=rows[0];d=rows[1]
        self.assertEqual((n['true_positive'],n['false_positive'],n['false_negative'],n['true_negative']),(1,1,1,0))
        self.assertEqual(n['f1'],.5);self.assertEqual(n['false_negative_rate'],.5)
        self.assertEqual(d['f1'],1.);self.assertEqual(d['false_positive'],0)
        self.assertEqual(len(rows),8);self.assertEqual(rows[2]['f1'],0.)

    def test_invalid_shapes_and_nonfinite_predictions_are_rejected(self):
        for y,p in [(np.zeros(8),np.zeros(8)),(np.zeros((2,8)),np.zeros((2,7))),(np.zeros((2,8)),np.full((2,8),np.nan)),(np.full((2,8),2),np.zeros((2,8)))]:
            with self.assertRaises(ValueError):confusion_rows(y,p)

    def test_resume_history_requires_exact_prefix_next_epoch_and_saved_lr(self):
        from report_controlled_results import verify_history_continuation
        previous=[{'epoch':16,'global_step':4560,'learning_rate':.00004,'next_learning_rate':.00003}]
        continuation=previous+[{'epoch':17,'global_step':4845,'learning_rate':.00003,'next_learning_rate':.00002}]
        verify_history_continuation(previous,continuation)
        for epoch,lr in ((16,.00003),(18,.00003),(17,.0001)):
            altered=previous+[{'epoch':epoch,'global_step':4845,'learning_rate':lr,'next_learning_rate':.00002}]
            with self.assertRaises(ValueError):verify_history_continuation(previous,altered)
