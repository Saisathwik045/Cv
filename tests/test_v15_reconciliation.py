"""Synthetic-only tests; no account, real labels, images or manifests."""
import copy
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from trustretina_v15 import reconciliation as r


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.root.chmod(0o700)
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir(mode=0o700)
        self.output = self.root / 'output'
        self.rows = {}
        for role in r.ROLES:
            rows = []
            # Deliberately non-sorted patient order; original allocator order wins.
            for suffix in ('z', 'a'):
                pid = f'fixture-{role}-{suffix}'
                for eye in ('left', 'right'):
                    rows.append({'patient_id': pid, 'eye': eye, 'labels': [1, 0, 0, 0, 0, 0, 0, 0],
                                 'component_id': r.canonical([pid]),
                                 'path': f'{pid}-{eye}-é.jpg', 'raw_sha256': 'a' * 64})
            self.rows[role] = rows
        self.expected = {'class_order': list(r.CLASSES), 'patient_counts': dict.fromkeys(r.ROLES, 2),
                         'sources': {role: {} for role in r.ROLES},
                         'expected': {role: r.digest(r.serialize(role, rows)) for role, rows in self.rows.items()}}
        self.save()

    def save(self):
        for role, rows in self.rows.items():
            data = json.dumps(rows, sort_keys=True).encode()
            path = self.inputs / f'{role}-rows.json'
            path.write_bytes(data)
            path.chmod(0o600)
            self.expected['sources'][role]['sha256'] = r.digest(data)

    def run_core(self):
        return r.reconcile(self.inputs, self.output, self.expected)

    def blocked(self):
        with self.assertRaises(r.Blocked):
            self.run_core()
        self.assertFalse((self.output / 'manifests').exists())
        if self.output.exists():
            evidence = self.output / 'reconciliation.json'
            if evidence.exists():
                text = evidence.read_text()
                self.assertNotIn('fixture-', text)
                self.assertNotIn('"labels":', text)
                self.assertEqual(json.loads(text)['status'], 'blocked')

    def test_exact_original_publisher_bytes(self):
        from trustretina.data.contained_release import publish_release
        inventory = {'data_kind': 'original_odir', 'patients': []}  # Synthetic publisher input only.
        partitions, groups = {}, []
        for role, rows in self.rows.items():
            partitions[role] = []
            for i in range(0, len(rows), 2):
                row = rows[i]
                partitions[role].append(row['patient_id'])
                groups.append([row['patient_id']])
                inventory['patients'].append({'patient_id': row['patient_id'], 'labels': row['labels'],
                    'eyes': {item['eye']: {'path': item['path'], 'sha256': item['raw_sha256']}
                             for item in rows[i:i+2]}})
        original = self.root / 'synthetic-publisher'
        publish_release(original, {'partitions': partitions, 'patient_groups': groups},
                        {'checks_passed': True, 'scheduled_missing': []}, inventory, provenance={})
        self.run_core()
        for role in r.ROLES:
            actual = (self.output / 'manifests' / f'{role}.json').read_bytes()
            self.assertEqual(actual, (original / f'{role}.json').read_bytes())
            self.assertFalse(actual.endswith(b'\n'))
            self.assertIn(b'\\u00e9', actual)
            self.assertEqual(r.digest(actual), self.expected['expected'][role])

    def test_private_permissions_under_permissive_umask(self):
        previous = os.umask(0)
        try:
            self.run_core()
        finally:
            os.umask(previous)
        for path in (self.output, *self.output.rglob('*')):
            self.assertEqual(path.stat().st_mode & 0o777, 0o700 if path.is_dir() else 0o600)

    def test_each_input_hash_mismatch_before_any_parse(self):
        for role in r.ROLES:
            with self.subTest(role=role), tempfile.TemporaryDirectory(dir=self.root) as out:
                self.output = Path(out) / 'result'
                old = self.expected['sources'][role]['sha256']
                self.expected['sources'][role]['sha256'] = '0' * 64
                with patch.object(r.json, 'loads', side_effect=AssertionError('must not parse')):
                    with self.assertRaisesRegex(r.Blocked, 'input_hash_mismatch'):
                        self.run_core()
                self.expected['sources'][role]['sha256'] = old
                self.assertFalse((self.output / 'manifests').exists())

    def test_patient_order_is_never_repaired(self):
        self.rows['train'] = self.rows['train'][2:] + self.rows['train'][:2]
        self.save()
        self.blocked()
        report = json.loads((self.output / 'reconciliation.json').read_text())
        self.assertEqual(report['failure_code'], 'original_manifest_hash_mismatch')
        self.assertEqual(set(report['output_sha256']), set(r.ROLES))

    def test_eye_order(self):
        self.rows['train'][0:2] = reversed(self.rows['train'][0:2]); self.save(); self.blocked()

    def test_duplicate_patient(self):
        self.rows['train'][2:] = copy.deepcopy(self.rows['train'][:2]); self.save(); self.blocked()

    def test_missing_eye(self):
        self.rows['train'].pop(); self.save(); self.blocked()

    def test_class_order(self):
        self.expected['class_order'].reverse(); self.blocked()

    def test_eye_label_mismatch(self):
        self.rows['calibration'][0]['labels'][0] = 0; self.save(); self.blocked()

    def test_consistently_modified_labels_fail_original_digest(self):
        for row in self.rows['calibration'][:2]:
            row['labels'] = [0, 1, 0, 0, 0, 0, 0, 0]
        self.save(); self.blocked()

    def test_nonbinary_labels(self):
        self.rows['train'][0]['labels'][0] = 2; self.save(); self.blocked()

    def test_eye_component_mismatch(self):
        self.rows['train'][0]['component_id'] = 'b' * 64; self.save(); self.blocked()

    def test_consistently_wrong_component(self):
        for row in self.rows['train'][:2]:
            row['component_id'] = 'b' * 64
        self.save(); self.blocked()

    def test_mixed_label_component_preserves_each_patient(self):
        for role, rows in self.rows.items():
            component = r.canonical(sorted({row['patient_id'] for row in rows}))
            for i, row in enumerate(rows):
                row['component_id'] = component
                if i >= 2:
                    row['labels'] = [0, 1, 0, 0, 0, 0, 0, 0]
            self.expected['expected'][role] = r.digest(r.serialize(role, rows))
        self.save(); self.run_core()
        self.assertEqual(json.loads((self.output / 'manifests/train.json').read_text())['rows'], self.rows['train'])

    def test_wrong_role_assignment(self):
        self.rows['validation'], self.rows['calibration'] = self.rows['calibration'], self.rows['validation']
        self.save(); self.blocked()

    def test_explicit_wrong_role(self):
        self.rows['train'][0]['role'] = 'final_evaluation'; self.save(); self.blocked()

    def test_cross_role_patient_component(self):
        self.rows['validation'] = copy.deepcopy(self.rows['train']); self.save(); self.blocked()

    def test_missing_input(self):
        (self.inputs / 'calibration-rows.json').unlink(); self.blocked()

    def test_world_readable_input(self):
        (self.inputs / 'train-rows.json').chmod(0o644); self.blocked()

    def test_world_readable_directory(self):
        self.inputs.chmod(0o755); self.blocked()

    def test_symlink_input(self):
        path = self.inputs / 'train-rows.json'
        path.rename(self.inputs / 'other'); path.symlink_to('other'); self.blocked()

    def test_symlink_output_parent(self):
        alias = self.root / 'alias'; alias.symlink_to(self.inputs, target_is_directory=True)
        self.output = alias / 'result'; self.blocked()

    def test_no_overwrite(self):
        self.run_core()
        with self.assertRaises(FileExistsError):
            self.run_core()

    def test_one_wrong_expected_digest_publishes_no_roles(self):
        self.expected['expected']['calibration'] = '0' * 64; self.blocked()
        self.assertEqual(list(self.output.glob('*.json')), [self.output / 'reconciliation.json'])

    def test_public_plan_offline(self):
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')):
            plan = r.plan()
        self.assertEqual(plan['sources']['train']['kernel_version'],
                         'saisathwikregula/trustretina-res-v13-inputs-767673ae/1')
        self.assertEqual(set(plan['expected']), set(r.ROLES))
        self.assertNotEqual(plan['sources']['train']['sha256'], plan['expected']['train'])

    def test_no_approval_blocks_before_private_access(self):
        with patch.object(r, 'reconcile', side_effect=AssertionError('private access')):
            with self.assertRaisesRegex(r.Blocked, 'approval_required'):
                r.authorized_reconcile('/missing', '/missing/output', {})

    def test_duplicate_json_keys(self):
        path = self.inputs / 'train-rows.json'
        data = b'[{"patient_id":"fixture-a","patient_id":"fixture-b"}]'
        path.write_bytes(data); self.expected['sources']['train']['sha256'] = r.digest(data)
        self.blocked()

    def test_staged_corruption_never_publishes_bundle(self):
        original = r.private_write
        def corrupt(path, data):
            original(path, data + b'\n' if path.name == 'validation.json' else data)
        with patch.object(r, 'private_write', side_effect=corrupt):
            self.blocked()
        self.assertEqual(json.loads((self.output / 'reconciliation.json').read_text())['failure_code'],
                         'staged_file_hash_mismatch')

    def test_write_failure_never_publishes_bundle(self):
        original = r.private_write
        def fail(path, data):
            if path.name == 'calibration.json':
                raise OSError('fixture-patient secret')
            return original(path, data)
        with patch.object(r, 'private_write', side_effect=fail):
            self.blocked()


if __name__ == '__main__':
    unittest.main()
