"""Bounded Mac recovery fixtures; no credentials, SDK activation or private objects."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('demo_recovery', ROOT / 'scripts/recover_inference_bundle.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class Raw:
    def __init__(self, data, fail=False):
        self.data, self.fail, self.calls = io.BytesIO(data), fail, 0

    def read(self, amount, decode_content=False):
        self.calls += 1
        if self.fail:
            raise OSError('fixture disconnect; never a real URL')
        return self.data.read(amount)


class Response:
    def __init__(self, data, status=200, fail=False):
        self.status_code = status
        self.headers = {'Content-Length': str(len(data))}
        self.raw = Raw(data, fail)
        self.closed = False

    def close(self):
        self.closed = True


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.destination = self.base / 'private/bundle'
        self.payloads = {name: ('fixture-' + name).encode() for name in recovery.SIZES}
        self.contract = self.base / 'candidate-contract.json'
        self.contract.write_bytes(b'public synthetic contract')
        objects = [{'name': n, 'remote': 'full/' + n,
                    'version': recovery.VERSIONS[0 if n == 'best.pt' else 1],
                    'size': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
                   for n, b in self.payloads.items()]
        self.plan = {'objects': objects, 'contract_sha256': recovery.digest(self.contract),
                     'expected_private_bytes': sum(len(b) for b in self.payloads.values()),
                     'maximum_transfer_bytes': 2 * sum(len(b) for b in self.payloads.values()),
                     'minimum_free_bytes': 0, 'maximum_attempts_per_object': 2,
                     'download_hosts': [recovery.HOST]}
        self.responses = []
        self.urls = []
        def url(version, remote):
            self.urls.append((version, remote))
            return 'https://' + recovery.HOST + '/' + remote
        def open_url(url):
            response = Response(self.payloads[url.rsplit('/', 1)[-1]])
            self.responses.append(response)
            return response
        self.transport = SimpleNamespace(url=url, open=open_url)

    def run_recovery(self, **kwargs):
        return recovery.recover(self.destination, approved=True, plan=self.plan,
                                transport_factory=lambda: self.transport,
                                contract_path=self.contract, verifier=self.verify, **kwargs)

    def verify(self, folder):
        self.assertEqual({p.name for p in folder.iterdir()}, {*self.payloads, 'candidate-contract.json'})
        for obj in self.plan['objects']:
            self.assertEqual(recovery.digest(folder / obj['name']), obj['sha256'])
        self.assertEqual(recovery.digest(folder / 'candidate-contract.json'), self.plan['contract_sha256'])

    def state(self):
        return json.loads((self.destination.parent / '.bundle-recovery/state.json').read_text())

    def test_production_plan_pins_four_objects_sizes_and_frozen_hashes(self):
        p = recovery.build_plan()
        self.assertEqual(len(p['objects']), 4)
        self.assertEqual(p['expected_private_bytes'], 334241994)
        self.assertEqual(p['maximum_transfer_bytes'], 668483988)
        record = json.loads((ROOT / 'results/demo-v8/artifact-recovery.json').read_text())
        for o in p['objects']:
            self.assertIn(o['version'], recovery.VERSIONS)
            self.assertEqual(o['sha256'], record['files'][o['name']])
        self.assertEqual(p['contract_sha256'], record['files']['candidate-contract.json'])

    def test_explicit_gate_precedes_storage_or_transport(self):
        factory = Mock()
        with self.assertRaisesRegex(recovery.RecoveryError, 'allow-recovery'):
            recovery.recover(self.destination, transport_factory=factory)
        factory.assert_not_called()
        self.assertFalse(self.destination.parent.exists())

    def test_success_hashes_private_permissions_and_receipt_without_model_loading(self):
        result = self.run_recovery()
        self.assertEqual(result['known_transferred_bytes'], self.plan['expected_private_bytes'])
        self.assertEqual(result['conservatively_charged_bytes'], result['known_transferred_bytes'])
        self.assertFalse(result['checkpoint_loaded'])
        self.assertEqual(self.urls, [(o['version'], o['remote']) for o in self.plan['objects']])
        self.assertTrue(all(r.closed for r in self.responses))
        for p in [self.destination.parent, *self.destination.parent.rglob('*')]:
            self.assertFalse(p.is_symlink())
            self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o700 if p.is_dir() else 0o600)
        self.verify(self.destination)

    def test_existing_verified_bundle_is_reused_without_authentication(self):
        self.run_recovery()
        with patch.object(self.transport, 'url', side_effect=AssertionError('network')):
            self.assertEqual(self.run_recovery()['status'], 'already_verified')

    def test_corrupt_existing_bundle_is_preserved(self):
        self.run_recovery()
        p = self.destination / 'best.pt'
        p.write_bytes(b'tampered')
        with self.assertRaises(AssertionError):
            self.run_recovery()
        self.assertEqual(p.read_bytes(), b'tampered')

    def test_symlink_or_insecure_private_directory_rejected(self):
        self.destination.parent.mkdir(mode=0o700)
        self.destination.symlink_to(self.base)
        with self.assertRaisesRegex(recovery.RecoveryError, 'Symlink'):
            self.run_recovery()
        self.destination.unlink()
        self.destination.parent.chmod(0o755)
        with self.assertRaisesRegex(recovery.RecoveryError, 'permissions'):
            self.run_recovery()

    def test_disk_gate_precedes_credentials(self):
        with patch.object(recovery.shutil, 'disk_usage', return_value=SimpleNamespace(free=-1)):
            with self.assertRaisesRegex(recovery.RecoveryError, 'free space'):
                self.run_recovery()
        self.assertFalse(self.urls)

    def test_redirect_and_wrong_size_never_read_body(self):
        for kind in ('redirect', 'size', 'encoding'):
            response = Response(b'fixture')
            if kind == 'redirect':
                response.status_code = 302
            if kind == 'size':
                response.headers['Content-Length'] = '9999999'
            if kind == 'encoding':
                response.headers['Content-Encoding'] = 'gzip'
            obj = {'size': 7, 'sha256': hashlib.sha256(b'fixture').hexdigest()}
            state = {'charged_bytes': 0, 'maximum_transfer_bytes': 100, 'known_transferred_bytes': 0}
            with self.assertRaises(recovery.RecoveryError):
                recovery.receive(response, self.base / kind, obj, state, lambda: None)
            self.assertEqual(response.raw.calls, 0)

    def test_bad_hash_blocks_publication_and_future_attempts(self):
        self.payloads['best.pt'] = b'x' * self.plan['objects'][0]['size']
        with self.assertRaisesRegex(recovery.RecoveryError, 'SHA256'):
            self.run_recovery()
        self.assertFalse(self.destination.exists())
        self.assertTrue(self.state()['blocked'])
        self.transport.open = Mock(side_effect=AssertionError('no retry'))
        with self.assertRaisesRegex(recovery.RecoveryError, 'requires review'):
            self.run_recovery()

    def test_interrupted_read_remains_charged_and_retry_is_bounded(self):
        size = self.plan['objects'][0]['size']
        self.transport.open = lambda _: Response(b'x' * size, fail=True)
        for _ in range(2):
            with self.assertRaises(OSError):
                self.run_recovery()
        self.assertEqual(self.state()['charged_bytes'], size * 2)
        self.assertEqual(self.state()['known_transferred_bytes'], 0)
        with self.assertRaisesRegex(recovery.RecoveryError, 'retry limit'):
            self.run_recovery()

    def test_completed_inputs_are_reused_after_interruption(self):
        old_open = self.transport.open
        def fail_second(url):
            if url.endswith('temperature.json'):
                raise OSError('fixture disconnect')
            return old_open(url)
        self.transport.open = fail_second
        with self.assertRaises(OSError):
            self.run_recovery()
        self.transport.open = old_open
        self.urls.clear()
        self.run_recovery()
        self.assertNotIn((self.plan['objects'][0]['version'], 'full/best.pt'), self.urls)

    def test_cumulative_budget_prevents_read(self):
        response = Response(b'1234')
        state = {'charged_bytes': 8, 'known_transferred_bytes': 8, 'maximum_transfer_bytes': 10}
        with self.assertRaisesRegex(recovery.RecoveryError, 'budget'):
            recovery.receive(response, self.base / 'partial', {'size': 4, 'sha256': 'unused'}, state, lambda: None)
        self.assertEqual(response.raw.calls, 0)

    def test_hosts_and_credentials_in_urls_rejected(self):
        for url in ('http://' + recovery.HOST + '/x', 'https://other.invalid/x',
                    'https://user:secret@' + recovery.HOST + '/x'):
            with self.assertRaises(recovery.RecoveryError):
                recovery.validate_url(url)

    def test_dangling_metadata_symlink_is_preserved_and_rejected(self):
        p = self.base / 'state.json'
        p.symlink_to(self.base / 'absent')
        with self.assertRaisesRegex(recovery.RecoveryError, 'permissions'):
            recovery.write_json(p, {'charged_bytes': 0})
        self.assertTrue(p.is_symlink())
        self.assertFalse((self.base / 'absent').exists())

    def test_changed_plan_cannot_reset_an_interrupted_ledger(self):
        self.transport.open = Mock(side_effect=OSError('fixture interruption'))
        with self.assertRaises(OSError):
            self.run_recovery()
        self.plan['maximum_transfer_bytes'] += 1
        with self.assertRaisesRegex(recovery.RecoveryError, 'binding changed'):
            self.run_recovery()

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'optional app Pillow')
    def test_local_demo_generator_creates_private_non_patient_pngs_and_reuses_them(self):
        spec = importlib.util.spec_from_file_location('demo_images', ROOT / 'scripts/generate_demo_images.py')
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        folder = self.base / 'demo-images'
        generator.generate(folder)
        from PIL import Image
        hashes = {}
        for name in ('left.png', 'right.png', 'replacement.png'):
            p = folder / name
            self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
            with Image.open(p) as image:
                self.assertEqual((image.format, image.size, image.mode), ('PNG', (320, 240), 'RGB'))
            hashes[name] = recovery.digest(p)
        generator.generate(folder)
        self.assertEqual(hashes, {name: recovery.digest(folder / name) for name in hashes})

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'optional app Pillow')
    def test_demo_generator_never_overwrites_a_different_image(self):
        spec = importlib.util.spec_from_file_location('demo_images', ROOT / 'scripts/generate_demo_images.py')
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        folder = self.base / 'demo-images'
        folder.mkdir(mode=0o700)
        p = folder / 'left.png';p.write_bytes(b'existing fixture');p.chmod(0o600)
        with self.assertRaisesRegex(generator.RecoveryError, 'preserved'):
            generator.generate(folder)
        self.assertEqual(p.read_bytes(), b'existing fixture')

    def test_sdk_listing_pins_v1_exact_name_and_ignores_log(self):
        transport = recovery.KaggleTransport.__new__(recovery.KaggleTransport)
        transport.allowed = {(recovery.VERSIONS[0], 'full/best.pt')}
        method = Mock(return_value=SimpleNamespace(files=[SimpleNamespace(file_name='full/best.pt', url='https://' + recovery.HOST + '/selected')], next_page_token=''))
        client = SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(list_kernel_session_output=method)))
        context = Mock();context.__enter__ = Mock(return_value=client);context.__exit__ = Mock(return_value=False)
        transport.api = SimpleNamespace(build_kaggle_client=lambda: context)
        module = SimpleNamespace(ApiListKernelSessionOutputRequest=SimpleNamespace)
        with patch.dict(sys.modules, {'kagglesdk.kernels.types.kernels_api_service': module}):
            self.assertEqual(transport.url(recovery.VERSIONS[0], 'full/best.pt'), 'https://' + recovery.HOST + '/selected')
        request = method.call_args.args[0]
        self.assertEqual((request.version_label, request.page_size), ('v1', 100))
        with self.assertRaisesRegex(recovery.RecoveryError, 'scope'):
            transport.url(recovery.VERSIONS[0], 'full/last.pt')

    @unittest.skipUnless(importlib.util.find_spec('requests'), 'optional requests')
    def test_http_transport_disables_redirects_decoding_and_netrc(self):
        import requests
        with patch.object(requests, 'get') as get:
            recovery.KaggleTransport.__new__(recovery.KaggleTransport).open('https://' + recovery.HOST + '/selected')
        kw = get.call_args.kwargs
        self.assertFalse(kw['allow_redirects'])
        self.assertTrue(kw['stream'])
        self.assertEqual(kw['headers'], {'Accept-Encoding': 'identity'})
        self.assertIsInstance(kw['auth'], requests.auth.AuthBase)


if __name__ == '__main__':
    unittest.main()
