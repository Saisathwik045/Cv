import unittest
import numpy as np
try:
 import cv2,skimage
 AVAILABLE=True
except ImportError:AVAILABLE=False
from trustretina.data.bounded_review import select_detail_pairs,material_uncertainty,coarse_content,prepare_detail,compare_detail

class ReviewBudgetTests(unittest.TestCase):
 def test_deterministic_budget_mandatory_and_dual_ranking(self):
  pairs=[(0,i) for i in range(1,6)];scores=[(.2,.2),(.99,.3),(.5,.99),(.1,.1),(.8,.8)]
  self.assertEqual(select_detail_pairs(pairs,scores,[3],maximum=3),[3,1,2])
  with self.assertRaises(ValueError):select_detail_pairs(pairs,scores,[0,1],maximum=1)
 def test_screening_is_uncertainty_not_duplicate_proof(self):
  self.assertTrue(material_uncertainty(('a','b'),(.99,.5)))
  self.assertFalse(material_uncertainty(('a','a'),(.99,.99)))
  self.assertFalse(material_uncertainty(('a','b'),(.99,.99),{'link':True}))
  self.assertTrue(material_uncertainty(('a','b'),(.1,.1),prior_possible_copy=True))

@unittest.skipUnless(AVAILABLE,'Dedicated photographic CI requires optional dependencies')
class DetailedReviewTests(unittest.TestCase):
 def chart(self,seed=11):
  rng=np.random.default_rng(seed);rgb=np.zeros((640,640,3),np.uint8);yy,xx=np.mgrid[:640,:640];mask=(xx-320)**2+(yy-320)**2<290**2
  gray=np.clip(180-.2*np.hypot(xx-320,yy-320)+rng.normal(0,2,(640,640)),0,255).astype(np.uint8)
  rgb[mask]=np.stack([gray,gray//2,gray//3],-1)[mask]
  for _ in range(220):
   x,y=rng.integers(100,540,2);cv2.circle(rgb,(int(x),int(y)),int(rng.integers(2,8)),tuple(int(x) for x in rng.integers(20,220,3)),-1)
  return rgb
 def test_content_invariant_to_black_padding(self):
  a=self.chart();b=np.pad(a,((30,30),(40,40),(0,0)))
  self.assertGreater(float(np.dot(coarse_content(a)['intensity'],coarse_content(b)['intensity'])),.98)
 def test_independent_photographic_copy_evidence(self):
  a=self.chart();b=cv2.resize(a,(900,900));result=compare_detail(prepare_detail(a),prepare_detail(b))
  self.assertTrue(result['link'],result);self.assertGreaterEqual(result['inliers'],12)
 def test_different_detail_failure_not_fabricated_adjudication(self):
  result=compare_detail(prepare_detail(self.chart()),prepare_detail(self.chart(44)))
  self.assertFalse(result['link']);self.assertEqual(result['classification'],'unresolved_candidate')
 def test_failed_controls_and_low_information(self):
  a=self.chart();b=cv2.resize(a,(800,800));result=compare_detail(prepare_detail(a),prepare_detail(b),controls_passed=False)
  self.assertFalse(result['link'])
  result=compare_detail(prepare_detail(np.ones((512,512,3),np.uint8)*50),prepare_detail(np.ones((512,512,3),np.uint8)*51))
  self.assertEqual(result['classification'],'unresolved_candidate')
