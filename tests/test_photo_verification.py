"""Independent photo evidence on synthetic pixels; never clinical/model fitting."""
import unittest
import numpy as np
try:
 import cv2
 import skimage
 AVAILABLE=True
except ImportError:
 AVAILABLE=False
from trustretina.data.photo_verification import prepare_photo,verify_photos

@unittest.skipUnless(AVAILABLE,'Install duplicate-review extra for photographic controls')
class PhotoEvidenceTests(unittest.TestCase):
 def chart(self,seed=11):
  rng=np.random.default_rng(seed);image=np.zeros((384,384,3),np.uint8)
  yy,xx=np.mgrid[:384,:384];inside=(xx-192)**2+(yy-192)**2<=175**2
  gradient=np.clip(150-.25*np.sqrt((xx-192)**2+(yy-192)**2)+rng.normal(0,2,(384,384)),0,255).astype(np.uint8)
  image[inside]=np.stack([gradient,gradient//2,gradient//3],axis=-1)[inside]
  for i in range(100):
   x,y=rng.integers(50,330,2);cv2.circle(image,(int(x),int(y)),int(rng.integers(2,7)),tuple(int(v) for v in rng.integers(20,220,3)),-1)
  return image
 def test_exact_pixels_confirmed(self):
  image=self.chart();a=prepare_photo(image);b=prepare_photo(image.copy())
  self.assertEqual(verify_photos(a,b)['classification'],'confirmed_duplicate')
 def test_resize_jpeg_known_copy_has_independent_spatial_evidence(self):
  image=self.chart();small=cv2.resize(image,(300,300));_,encoded=cv2.imencode('.jpg',cv2.cvtColor(small,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90]);decoded=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
  evidence=verify_photos(prepare_photo(image),prepare_photo(decoded))
  self.assertEqual(evidence['classification'],'strongly_supported_probable_copy',evidence)
 def test_low_information_not_confirmed_by_similar_background(self):
  a=prepare_photo(np.full((256,256,3),100,np.uint8));b=prepare_photo(np.full((256,256,3),101,np.uint8))
  evidence=verify_photos(a,b,distinct_visual_review=True)
  self.assertEqual(evidence['classification'],'unresolved_candidate');self.assertFalse(evidence['link'])
 def test_failed_match_alone_is_not_false_positive(self):
  evidence=verify_photos(prepare_photo(self.chart(11)),prepare_photo(self.chart(99)))
  self.assertEqual(evidence['classification'],'unresolved_candidate')
 def test_failed_controls_disable_probable_copy_link(self):
  image=self.chart();other=cv2.resize(image,(300,300))
  evidence=verify_photos(prepare_photo(image),prepare_photo(other),controls_passed=False)
  self.assertFalse(evidence['link']);self.assertEqual(evidence['classification'],'unresolved_candidate')
