import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_development_v12 import ARMS, counts, outcomes, pixel_features, select_cases, positive_support_summary, pinned_output


class DevelopmentAuditTests(unittest.TestCase):
    def test_source_metadata_requires_report_receipt_and_output_hashes(self):
        import hashlib,json,tempfile
        with tempfile.TemporaryDirectory() as temp:
            b=Path(temp);(b/'download').mkdir();target=b/'download/train-only.json';target.write_text('{}')
            identity={'nonce':'fixture','source_commit':'fixture'}
            report={**identity,'output_hashes':{'train-only.json':hashlib.sha256(target.read_bytes()).hexdigest()}}
            rp=b/'download/report.json';rp.write_text(json.dumps(report))
            (b/'submission.json').write_text(json.dumps(identity))
            (b/'verification.json').write_text(json.dumps({**identity,'integrity_verified':True,'artifact_sha256':hashlib.sha256(rp.read_bytes()).hexdigest()}))
            self.assertEqual(pinned_output(b,'report.json','train-only.json'),target)
            target.write_text('{"tampered":true}')
            with self.assertRaisesRegex(ValueError,'output mismatch'):pinned_output(b,'report.json','train-only.json')

    def test_positive_support_only_reports_recall_not_conditional_precision(self):
        self.assertEqual(positive_support_summary(np.asarray(['TP','FN'])),{'n':2,'TP':1,'FN':1,'recall':.5})
        with self.assertRaises(ValueError):positive_support_summary(np.asarray(['FP']))

    def test_counts_keep_multilabel_threshold_boundary_and_errors(self):
        status = outcomes([1, 1, 0, 0], [.5, .49, .7, .3])
        self.assertEqual(status.tolist(), ['TP', 'FN', 'FP', 'TN'])
        self.assertEqual(counts(status)['f1'], .5)
        with self.assertRaises(ValueError):
            outcomes([1], [float('nan')])

    def test_selection_is_order_invariant_and_component_distinct(self):
        ids = [str(i) for i in range(32)]
        groups = ['group'+str(i//2) for i in range(32)]
        y = np.zeros((32,8),dtype=int);p = np.full((32,8),.2)
        for j in (5,7):
            y[:16,j] = 1;p[:8,j] = .7;p[16:24,j] = .7
        def keyed(records, identity):
            return sorted((r['arm'],r['label'],r['outcome'],identity[r['index']]) for r in records)
        original = select_cases(ids,groups,y,{a:p for a in ARMS})
        order = np.arange(31,-1,-1)
        flipped = select_cases([ids[i] for i in order],[groups[i] for i in order],y[order],{a:p[order] for a in ARMS})
        self.assertEqual(keyed(original,ids),keyed(flipped,[ids[i] for i in order]))
        self.assertEqual(len(original),32)
        for arm in ARMS:
            for label in ('H','O'):
                for outcome in ('TP','FP','FN','TN'):
                    selected=[r for r in original if (r['arm'],r['label'],r['outcome'])==(arm,label,outcome)]
                    self.assertEqual(len({groups[r['index']] for r in selected}),2)

    def test_no_refill_when_stratum_missing(self):
        y=np.zeros((3,8),int);p=np.zeros((3,8))
        records=select_cases(['a','b','c'],['g','g','g'],y,{a:p for a in ARMS})
        self.assertEqual(len(records),4)
        self.assertTrue(all(r['outcome']=='TN' for r in records))

    def test_padding_and_dark_edges_do_not_create_gradient_proxy(self):
        x=np.zeros((224,224,3),dtype=np.uint8);x[:20]=[124,116,104];x[60:180,60:180]=100
        q=pixel_features(x)
        self.assertAlmostEqual(q['mean_color_fraction'],20/224)
        self.assertEqual(q['foreground_gradient_energy'],0)
        self.assertAlmostEqual(q['foreground_contrast'],0)
        self.assertIsNone(pixel_features(np.zeros_like(x))['foreground_gradient_energy'])
        with self.assertRaises(ValueError):pixel_features(x.astype(float))


if __name__=='__main__':unittest.main()
