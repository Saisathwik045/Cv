"""Meaningful input-pair checks; all images are generated non-patient fixtures."""
import unittest
from PIL import Image
from dashboard.uploads import declared_eye, pair_errors


class PairValidationTests(unittest.TestCase):
    def test_declared_side_and_matching_id(self):
        self.assertEqual(declared_eye('123_left.JPG'), ('left', '123'))
        self.assertEqual(declared_eye('pair01-RIGHT.png'), ('right', 'pair01'))
        self.assertEqual(declared_eye('right-eye.jpg'), ('right', None))
        self.assertEqual(declared_eye('photograph.png'), (None, None))
        self.assertEqual(pair_errors('123_left.jpg', '123_right.jpg', [b'a', b'b'], [None, None]), [])

    def test_wrong_slots_and_different_declared_patients(self):
        issues = pair_errors('123_right.jpg', '456_left.jpg', [b'a', b'b'], [None, None])
        self.assertEqual(len(issues), 3)

    def test_exact_duplicates_even_when_different_encodings_or_names(self):
        image = Image.new('RGB', (16, 16), (10, 20, 30))
        self.assertTrue(pair_errors('a.png', 'b.png', [b'a', b'b'], [image, image.copy()]))
        self.assertTrue(pair_errors('a.png', 'b.png', [b'a', b'a'], [None, None]))

    def test_unknown_names_do_not_claim_verified_laterality(self):
        self.assertEqual(pair_errors('scan1.png', 'scan2.png', [b'a', b'b'], [None, None]), [])

    def test_replacing_pair_resets_operator_confirmation(self):
        from dashboard.uploads import reconcile_confirmation
        state = {}
        reconcile_confirmation(state, ('a', 'b'))
        state['pair_confirmed'] = True
        reconcile_confirmation(state, ('a', 'b'))
        self.assertTrue(state['pair_confirmed'])
        reconcile_confirmation(state, ('a', 'new'))
        self.assertNotIn('pair_confirmed', state)
