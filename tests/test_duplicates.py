"""Duplicate containment on synthetic pixels/patient annotations, never real data."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from synthetic import make_fixture
from trustretina.data.duplicates import POLICY, components
from trustretina.data.validation import DataValidationError, validate_dataset, file_hash
from trustretina.data.splits import prepare_data, make_splits, audit_manifest


class DuplicateGroupsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = make_fixture(self.temp.name)
        self.config['data']['duplicate_policy'] = POLICY
        self.root = Path(self.config['data']['root'])
        self.images = self.root / 'Training Images'
        self.output = Path(self.temp.name) / 'output'

    def connect(self, a, b):
        shutil.copyfile(self.images / a, self.images / b)

    def transitive_inventory(self):
        self.connect('0_left.png', '1_left.png')
        self.connect('1_right.png', '2_right.png')
        return validate_dataset(self.config)

    def test_transitive_connections_keep_both_eyes_and_conflicting_annotations(self):
        inventory = self.transitive_inventory()
        adjudication = inventory['duplicate_adjudication']
        self.assertIn(['0', '1', '2'], adjudication['patient_groups'])
        self.assertEqual(adjudication['counts']['candidate_pairs'], 2)
        self.assertEqual(adjudication['counts']['confirmed_exact_pairs'], 2)
        self.assertEqual(adjudication['counts']['conflicting_annotation_pairs'], 2)
        self.assertTrue(all(p['decision'] == 'co_group' for p in adjudication['pairs']))
        self.assertEqual([p['labels'] for p in inventory['patients'] if p['patient_id'] in ('0', '1', '2')],
                         [[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0]])
        manifest = make_splits(inventory, self.config)
        locations = {p['patient_id']: name for name, rows in manifest['partitions'].items() for p in rows}
        self.assertEqual(len({locations[pid] for pid in ('0', '1', '2')}), 1)
        self.assertTrue(all(set(p['eyes']) == {'left', 'right'} for rows in manifest['partitions'].values() for p in rows))
        self.assertEqual(sum(len(rows) for rows in manifest['partitions'].values()), 160)

    def test_perceptual_candidate_is_unresolved_and_contained_not_declared_duplicate(self):
        from PIL import Image
        with Image.open(self.images / '0_left.png') as image:
            image.putpixel((0, 0), (0, 0, 0))
            image.save(self.images / '1_left.png')
        inventory = validate_dataset(self.config)
        pair = inventory['duplicate_adjudication']['pairs'][0]
        self.assertFalse(pair['file_equal'])
        self.assertFalse(pair['pixel_equal'])
        self.assertEqual(pair['classification'], 'unresolved_perceptual_candidate')
        self.assertEqual(pair['decision'], 'co_group')
        self.assertIn(['0', '1'], inventory['duplicate_adjudication']['patient_groups'])
        self.assertFalse(pair['clinical_adjudication'])
        self.assertIn('rgb_mae_64', pair['image_comparison'])

    def test_audit_rejects_cross_partition_edge_even_with_complete_unique_patients(self):
        inventory = self.transitive_inventory()
        manifest = make_splits(inventory, self.config)
        origin = next(name for name, rows in manifest['partitions'].items() if any(p['patient_id']=='0' for p in rows))
        target = next(name for name in manifest['partitions'] if name != origin)
        row = next(p for p in manifest['partitions'][origin] if p['patient_id']=='0')
        manifest['partitions'][origin].remove(row)
        manifest['partitions'][target].append(row)
        with self.assertRaises(DataValidationError):
            audit_manifest(manifest, inventory)

    def test_perceptual_only_edge_cannot_cross_partitions(self):
        from PIL import Image
        with Image.open(self.images / '0_left.png') as image:
            image.putpixel((0, 0), (0, 0, 0))
            image.save(self.images / '1_left.png')
        inventory = validate_dataset(self.config)
        manifest = make_splits(inventory, self.config)
        origin = next(name for name, rows in manifest['partitions'].items()
                      if any(p['patient_id'] == '0' for p in rows))
        target = next(name for name in manifest['partitions'] if name != origin)
        row = next(p for p in manifest['partitions'][origin] if p['patient_id'] == '0')
        manifest['partitions'][origin].remove(row)
        manifest['partitions'][target].append(row)
        with self.assertRaisesRegex(DataValidationError, 'Duplicate-connected'):
            audit_manifest(manifest, inventory)

    def test_audit_cannot_hide_edges_by_tampering_adjudication(self):
        inventory = self.transitive_inventory()
        for field in ('pairs', 'patient_groups'):
            altered = copy.deepcopy(inventory)
            altered['duplicate_adjudication'][field] = []
            with self.assertRaisesRegex(DataValidationError, 'adjudication'):
                make_splits(altered, self.config)

    def test_indivisible_positive_class_component_fails_without_publication(self):
        for pid in range(8, 160, 8):
            self.connect('0_left.png', f'{pid}_left.png')
        with self.assertRaisesRegex(DataValidationError, 'grouping.*infeasible'):
            prepare_data(self.config, self.output)
        self.assertFalse(self.output.exists())

    def test_same_patient_duplicate_retained_and_exact_pixels_reencoded_confirmed(self):
        from PIL import Image
        self.connect('0_left.png','0_right.png')
        with Image.open(self.images/'0_left.png') as image:
            image.save(self.images/'1_left.png',format='BMP')
        inventory = validate_dataset(self.config)
        self.assertEqual(inventory['duplicate_adjudication']['counts']['candidate_pairs'],3)
        pairs = inventory['duplicate_adjudication']['pairs']
        self.assertTrue(all(p['pixel_equal'] for p in pairs))
        self.assertTrue(any(not p['file_equal'] for p in pairs))
        self.assertEqual(len(inventory['patients']),160)
        self.assertIn(['0','1'], inventory['duplicate_adjudication']['patient_groups'])

    def test_immutable_adjudication_and_deterministic_reuse(self):
        self.transitive_inventory()
        raw = {p: file_hash(p) for p in self.root.rglob('*') if p.is_file()}
        prepare_data(self.config, self.output)
        first = {p.name: file_hash(p) for p in self.output.iterdir()}
        self.assertTrue(prepare_data(self.config, self.output)['reused'])
        self.assertEqual(first,{p.name:file_hash(p) for p in self.output.iterdir()})
        other = self.output.with_name('independent')
        prepare_data(self.config, other)
        self.assertEqual(first,{p.name:file_hash(p) for p in other.iterdir()})
        self.assertEqual(raw,{p:file_hash(p) for p in raw})
        adjudication = self.output/'adjudication.json'
        adjudication.write_text('{}')
        with self.assertRaisesRegex(DataValidationError, 'checksum'):
            prepare_data(self.config, self.output)

    def test_transitive_graph_order_independence(self):
        patients = [{'patient_id':p} for p in ('a','b','c','d')]
        pairs = [{'a':{'patient_id':'a'},'b':{'patient_id':'b'}},
                 {'a':{'patient_id':'b'},'b':{'patient_id':'c'}}]
        self.assertEqual(components(patients,pairs),[['a','b','c'],['d']])
        self.assertEqual(components(patients,list(reversed(pairs))),[['a','b','c'],['d']])

    def test_checkpoint_cannot_mutate_live_labels_or_edges(self):
        self.connect('0_left.png', '1_left.png')
        expected = validate_dataset(self.config)
        stages = []
        def checkpoint(stage, value):
            stages.append(stage)
            value.clear()
        actual = validate_dataset(self.config, checkpoint=checkpoint)
        self.assertEqual(actual, expected)
        self.assertIn('image_inventory', stages)
        self.assertIn('candidate_adjudication', stages)
        self.assertIn('validated_inventory', stages)

    def test_cached_comparisons_preserve_all_numeric_evidence(self):
        from trustretina.data.duplicates import candidates, comparison, adjudicate
        self.connect('0_left.png','1_left.png')
        self.connect('0_left.png','2_left.png')
        inventory=validate_dataset(self.config)
        pairs=candidates(inventory['patients'],2)
        expected=[comparison(self.root,pair) for pair in pairs]
        actual=adjudicate(inventory['patients'],2,self.root)
        self.assertEqual(expected,[p['image_comparison'] for p in actual['pairs']])
