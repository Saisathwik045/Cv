"""Recovery fixtures only. No credentials, remote APIs or real artifact bytes."""
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from trustretina_v15 import recovery as r
from trustretina_v15.v15_fixture import create_fixture
from trustretina_v15.v15_inputs import audit_roles, align_role, load_npz, authorize as fit_authorize

ROOT = Path(__file__).resolve().parents[1]


class FakeTransport:
    def __init__(self, files):
        self.files = files; self.calls = []; self.fail = None; self.interrupt = None

    def download(self, version, name, target, limit):
        self.calls.append((version, name))
        data = self.files[version, name]
        with open(target, 'xb') as stream:
            target.chmod(0o600)
            if self.fail == name:
                stream.write(data[:3])
                print('PRIVATE-SIGNED-URL-SECRET')
                raise OSError('PRIVATE-SIGNED-URL-SECRET')
            stream.write(data[:limit+1])
        if self.interrupt == name:
            raise KeyboardInterrupt()


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.private = self.root/'private'
        path, _, self.fit_approval = create_fixture(self.root/'fixtures', 'a'*64)
        manifest = json.loads(path.read_text()); self.manifest = manifest
        registry = manifest['fixture_registry']; receipts = {}; self.files = {}
        # Match all 17 V14 registry names without any model checkpoint data.
        for fit in registry['fits']:
            folder = path.parent/fit['run_id']
            for name in ('last.pt', 'best.pt.sha256', 'last.pt.sha256', 'history.json', 'metrics.json', 'progress.json', 'dependencies.txt'):
                (folder/name).write_bytes(b'FIXTURE opaque integrity dependency')
                fit['artifact_sha256'][name] = r.digest(folder/name)
            stagepath = folder/'stage-report.json'; stage = json.loads(stagepath.read_text())
            stage['nonce'] = 'fixture-nonce-'+fit['run_id']
            stage['output_hashes'] = fit['artifact_sha256']
            stagepath.write_text(json.dumps(stage)); fit['report_sha256'] = r.digest(stagepath)
            receipts[fit['run_id']] = {'kernel_version': fit['kernel_version'], 'nonce': stage['nonce']}
            for name in ['stage-report.json', *fit['artifact_sha256']]:
                self.files[fit['kernel_version'], 'full/'+name] = (folder/name).read_bytes()
        frozen = {'receipts': receipts, 'registry': {'role_manifest_sha256': manifest['parent_manifest_sha256']}}
        self.plan = r.build_plan(registry, frozen, fixture=True)
        self.transport = FakeTransport(self.files)
        self.approval = {'approved': True, 'scope': r.FIXTURE_SCOPE, 'plan_sha256': r.identity(self.plan),
                         'private_root': str(self.private), 'allow_checkpoint_bytes': False,
                         'calibration_fitting_authorized': False, 'user_approval_reference': 'synthetic-test-only',
                         'maximum_file_bytes': 10_000_000, 'maximum_transfer_bytes': 100_000_000,
                         'download_hosts': ['storage.googleapis.com']}

    def recover(self):
        return r.recover_fixture(self.plan, self.approval, self.private, self.transport)

    def test_public_profiles_and_exact_loader_checkpoint_dependencies(self):
        metadata = r.public_plan(); full = r.public_plan('analysis')
        self.assertEqual(len(metadata['objects']), 32)
        self.assertEqual(len(full['objects']), 36)
        self.assertFalse(any(o['remote'].endswith('.pt') for o in metadata['objects']))
        self.assertEqual(sum(o['remote'].endswith('.pt') for o in full['objects']), 4)
        self.assertTrue(all(o['remote'].startswith('full/') for o in full['objects']))
        self.assertFalse(any('final' in o['remote'] or o['remote'].endswith(('.npy', '.png', '.jpg', '.log')) for o in full['objects']))
        from trustretina_v15 import v15_inputs
        self.assertEqual(r.REGISTRY_SHA256, v15_inputs.REGISTRY_SHA256)
        self.assertEqual(r.FROZEN_SHA256, v15_inputs.FROZEN_PLAN_SHA256)

    def test_exact_selection_hashes_modes_and_no_checkpoint_download(self):
        result = self.recover()
        self.assertEqual(result['verified_objects'], 32)
        self.assertTrue(result['recovery_complete'])
        self.assertFalse(result['role_alignment_verified'])
        self.assertFalse(result['calibration_fitting_authorized'])
        self.assertEqual(set(self.transport.calls), {(o['version'], o['remote']) for o in self.plan['objects']})
        self.assertEqual(self.private.stat().st_mode & 0o777, 0o700)
        for obj in self.plan['objects']:
            p = self.private/obj['local']
            self.assertEqual(r.digest(p), obj['sha256']); self.assertEqual(p.stat().st_mode & 0o777, 0o600)

    def test_duplicate_download_prevention_and_rehash_before_reuse(self):
        self.recover(); n = len(self.transport.calls); self.recover()
        self.assertEqual(len(self.transport.calls), n)
        target = self.private/self.plan['objects'][0]['local']; target.write_bytes(b'changed')
        with self.assertRaises(r.RecoveryError): self.recover()
        self.assertEqual(len(self.transport.calls), n)
        self.assertEqual(target.read_bytes(), b'changed')

    def test_missing_object_and_failed_download_leave_no_complete_marker(self):
        first = self.plan['objects'][0]
        del self.transport.files[first['version'], first['remote']]
        with self.assertRaises(r.RecoveryError): self.recover()
        self.assertFalse((self.private/'recovery-complete.json').exists())
        self.assertEqual(json.loads((self.private/'recovery-state.json').read_text())['status'], 'interrupted_or_blocked')

    def test_wrong_hash_is_preserved_and_not_published(self):
        first = self.plan['objects'][0]; self.transport.files[first['version'], first['remote']] = b'wrong bytes'
        with self.assertRaises(r.RecoveryError): self.recover()
        self.assertFalse((self.private/first['local']).exists())
        self.assertTrue(list((self.private/first['local']).parent.glob('*.partial-*')))

    def test_interrupted_complete_partial_is_reused_without_redownload(self):
        selected = self.plan['objects'][1]; self.transport.interrupt = selected['remote']
        with self.assertRaises(r.RecoveryError): self.recover()
        n = len(self.transport.calls); self.transport.interrupt = None
        self.recover()
        self.assertEqual(self.transport.calls.count((selected['version'], selected['remote'])), 1)
        self.assertEqual(len(self.transport.calls), 32)
        self.assertEqual(n, 2)

    def test_failed_partial_is_retained_and_only_incomplete_file_retried(self):
        selected = self.plan['objects'][1]; self.transport.fail = selected['remote']
        with self.assertRaises(r.RecoveryError): self.recover()
        self.transport.fail = None; self.recover()
        self.assertEqual(self.transport.calls.count((selected['version'], selected['remote'])), 2)
        first = self.plan['objects'][0]
        self.assertEqual(self.transport.calls.count((first['version'], first['remote'])), 1)
        self.assertTrue(list((self.private/selected['local']).parent.glob('*.partial-*')))

    def test_complete_recovery_missing_file_requires_review_without_refetch(self):
        self.recover(); (self.private/self.plan['objects'][0]['local']).unlink()
        n = len(self.transport.calls)
        with self.assertRaises(r.RecoveryError): self.recover()
        self.assertEqual(len(self.transport.calls), n)

    def test_wrong_version_or_untrusted_plan_rejected_before_transport(self):
        plan = r.public_plan(); plan['objects'][0]['version'] = plan['objects'][0]['version'][:-1]+'2'
        factory = Mock()
        with self.assertRaisesRegex(r.RecoveryError, 'pinned inventory'):
            r.recover(plan, None, self.private, transport_factory=factory)
        factory.assert_not_called(); self.assertFalse(self.private.exists())

    def test_tampered_recovery_plan_and_missing_approval_fail_before_filesystem(self):
        original = copy.deepcopy(self.approval)
        for field in ('approved', 'scope', 'plan_sha256', 'private_root', 'allow_checkpoint_bytes',
                      'calibration_fitting_authorized', 'user_approval_reference', 'maximum_file_bytes', 'download_hosts'):
            self.approval = {**original, field: None}
            with self.subTest(field=field), self.assertRaises(PermissionError): self.recover()
        self.assertFalse(self.private.exists()); self.assertFalse(self.transport.calls)

    def test_recovery_and_fitting_authorizations_are_not_interchangeable(self):
        self.approval = self.fit_approval
        with self.assertRaises(PermissionError): self.recover()
        with self.assertRaises(PermissionError):
            fit_authorize('a'*64, 'b'*64, {'scope': r.SCOPE}, fixture=False)

    def test_synthetic_permission_cannot_construct_real_transport(self):
        factory = Mock()
        with self.assertRaises(r.RecoveryError):
            r.recover(self.plan, self.approval, self.private, transport_factory=factory)
        factory.assert_not_called()

    def test_credential_absence_is_sanitized_after_recovery_permission_only(self):
        factory = Mock(side_effect=PermissionError('SECRET-TOKEN-VALUE'))
        with self.assertRaises(r.RecoveryError) as error:
            r._execute(self.plan, self.plan, self.approval, self.private, factory, fixture=True)
        self.assertNotIn('SECRET', str(error.exception)); self.assertEqual(factory.call_count, 1)
        self.assertFalse((self.private/'recovery-complete.json').exists())

    def test_provider_stdout_stderr_errors_are_never_exposed(self):
        self.transport.fail = self.plan['objects'][0]['remote']
        stdout = io.StringIO(); stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(r.RecoveryError) as error: self.recover()
        self.assertNotIn('SECRET', stdout.getvalue()+stderr.getvalue()+str(error.exception))
        self.assertNotIn('SECRET', (self.private/'recovery-state.json').read_text())

    def test_byte_budget_and_checkpoint_authorization(self):
        self.approval['maximum_file_bytes'] = 1
        with self.assertRaises(r.RecoveryError): self.recover()
        self.assertFalse((self.private/'recovery-complete.json').exists())
        self.assertFalse(any(p.suffix == '.pt' for p in self.private.rglob('*')))

    def test_checkpoint_bytes_require_explicit_analysis_profile_permission(self):
        self.plan['profile'] = 'analysis'
        for run in self.plan['runs']:
            for name in ('best.pt', 'last.pt'):
                self.plan['objects'].append({'version': run['version'], 'remote': 'full/'+name,
                    'local': run['run_id']+'/'+name, 'sha256': run['artifact_sha256'][name],
                    'purpose': 'opaque_checkpoint_hash_required_by_existing_v15_loader'})
        self.approval['plan_sha256'] = r.identity(self.plan)
        with self.assertRaises(PermissionError): self.recover()
        self.assertFalse(self.transport.calls)
        self.approval['allow_checkpoint_bytes'] = True
        self.assertEqual(self.recover()['verified_objects'], 36)
        self.assertFalse(self.recover()['calibration_fitting_authorized'])

    def test_retained_partial_bytes_count_toward_retry_budget(self):
        self.approval['maximum_transfer_bytes'] = 1
        with self.assertRaises(r.RecoveryError): self.recover()
        calls = len(self.transport.calls)
        with self.assertRaises(r.RecoveryError): self.recover()
        self.assertEqual(len(self.transport.calls), calls)

    def test_wrong_producer_revision_fails_even_when_transport_hash_matches(self):
        obj = self.plan['objects'][0]
        stage = json.loads(self.transport.files[obj['version'], obj['remote']])
        stage['source_commit'] = 'wrong-producer-revision'
        payload = json.dumps(stage).encode()
        self.transport.files[obj['version'], obj['remote']] = payload
        import hashlib
        obj['sha256'] = hashlib.sha256(payload).hexdigest()
        self.approval['plan_sha256'] = r.identity(self.plan)
        with self.assertRaises(r.RecoveryError): self.recover()
        self.assertFalse((self.private/'recovery-complete.json').exists())

    def test_public_result_has_no_patient_identifiers_or_fitting_permission(self):
        result = self.recover()
        self.assertNotIn('fixture-validation-', json.dumps(result))
        self.assertNotIn('fixture-calibration-', json.dumps(result))
        self.assertEqual(set(result), {'verified_objects', 'profile', 'data_kind', 'recovery_complete',
                                      'role_alignment_verified', 'calibration_fitting_authorized'})

    def test_private_root_permissions_symlinks_and_repository_paths(self):
        self.private.mkdir(); self.private.chmod(0o755)
        with self.assertRaises(r.RecoveryError): self.recover()
        self.private.chmod(0o700)
        (self.private/'recovery-state.json').symlink_to(self.root/'outside')
        with self.assertRaises(r.RecoveryError): self.recover()
        with self.assertRaises(r.RecoveryError): r.private_root(ROOT/'artifacts/recovery-forbidden')
        with self.assertRaises(r.RecoveryError): r.private_root('/tmp/../workspace/Cv/artifacts/recovery-forbidden')
        with self.assertRaises(r.RecoveryError): r.checked_path(self.private, '../escape')

    def test_role_patient_component_and_missing_eye_validation_remains_separate(self):
        self.recover()
        roles = {name: json.loads((self.root/'fixtures'/item['file']).read_text()) for name, item in self.manifest['roles'].items()}
        audit_roles(roles, fixture=True)
        original = copy.deepcopy(roles)
        for mutation in ('eye', 'component', 'patient'):
            roles = copy.deepcopy(original)
            if mutation == 'eye': roles['calibration'].pop()
            elif mutation == 'component':
                for row in roles['calibration'][:2]: row['component_id'] = roles['validation'][0]['component_id']
            else:
                for row in roles['calibration'][:2]: row['patient_id'] = roles['validation'][0]['patient_id']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): audit_roles(roles, fixture=True)
        # A byte-verified recovery never asserts patient alignment or fitting readiness.
        state = json.loads((self.private/'recovery-complete.json').read_text())
        self.assertFalse(state['role_alignment_verified']); self.assertFalse(state['v15_fitting_ready'])

    def test_prediction_component_corruption_rejected_by_existing_alignment_gate(self):
        self.recover(); run = self.plan['runs'][0]['run_id']; folder = self.private/run
        rows = json.loads((self.root/'fixtures/validation-rows.json').read_text())
        patients = load_npz(folder/'validation-patients.npz'); patients['component_ids'][0] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'alignment'):
            align_role(rows, load_npz(folder/'validation-images.npz'), patients,
                       load_npz(folder/'validation-passes.npz'), role='validation')

    def test_cli_plan_has_no_authentication_or_private_access(self):
        cmd = [sys.executable, '-B', str(ROOT/'scripts/recover_v15_artifacts.py')]
        result = subprocess.run(cmd+['plan'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(result.stdout)['plan']['objects']), 32)
        result = subprocess.run(cmd+['recover'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('separate approval', result.stderr)


class FakeResponse:
    def __init__(self, data=b'fixture bytes', *, status=200, headers=None):
        self.data = data; self.status_code = status; self.headers = headers or {}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self):
        if self.status_code >= 400: raise OSError('provider error with signed URL')
    def iter_content(self, chunk_size): yield self.data


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name)/'download.partial'
        obj = r.public_plan()['objects'][0]
        self.version, self.remote = obj['version'], obj['remote']
        self.transport = object.__new__(r.KaggleTransport)
        self.transport.hosts = ['storage.googleapis.com']
        self.transport.allowed = {(self.version, self.remote)}
        self.requests = []
        self.files = [types.SimpleNamespace(file_name=self.remote, url='https://storage.googleapis.com/fixture-only')]
        self.token = ''
        def listing(request):
            self.requests.append(request)
            return types.SimpleNamespace(files=self.files, next_page_token=self.token)
        client = types.SimpleNamespace(kernels=types.SimpleNamespace(kernels_api_client=types.SimpleNamespace(list_kernel_session_output=listing)))
        self.transport.api = types.SimpleNamespace(build_kaggle_client=lambda: contextlib.nullcontext(client))
        module = types.ModuleType('kagglesdk.kernels.types.kernels_api_service')
        module.ApiListKernelSessionOutputRequest = types.SimpleNamespace
        requests_module = types.ModuleType('requests')
        requests_module.get = Mock()
        requests_module.auth = types.SimpleNamespace(AuthBase=lambda: object())
        self.modules = patch.dict(sys.modules, {'kagglesdk.kernels.types.kernels_api_service': module,
                                              'requests': requests_module})
        self.modules.start(); self.addCleanup(self.modules.stop)

    def test_exact_file_and_version_label_stream_only_without_logs(self):
        self.files.append(types.SimpleNamespace(file_name='full/unrequested-file.npz', url='https://forbidden.invalid'))
        with patch('requests.get', return_value=FakeResponse()) as get:
            self.transport.download(self.version, self.remote, self.target, 100)
        self.assertEqual(self.requests[0].version_label, 'v1')
        self.assertEqual(self.requests[0].user_name, 'saisathwikregula')
        self.assertEqual(get.call_count, 1)
        sentinel = object()
        self.assertIs(get.call_args.kwargs['auth'](sentinel), sentinel)
        self.assertEqual(self.target.read_bytes(), b'fixture bytes')
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o600)

    def test_unapproved_object_missing_object_and_duplicate_listing_fail_closed(self):
        with self.assertRaises(r.RecoveryError):
            self.transport.download(self.version, 'full/final_evaluation.npz', self.target, 100)
        self.assertFalse(self.requests)
        for files in ([], [self.files[0], self.files[0]]):
            self.files = files
            with patch('requests.get') as get:
                with self.assertRaises(r.RecoveryError):
                    self.transport.download(self.version, self.remote, self.target, 100)
                get.assert_not_called()

    def test_signed_redirect_host_must_be_approved_before_next_request(self):
        with patch('requests.get', return_value=FakeResponse(status=302, headers={'Location': 'https://unapproved.invalid/secret'})) as get:
            with self.assertRaises(r.RecoveryError):
                self.transport.download(self.version, self.remote, self.target, 100)
            self.assertEqual(get.call_count, 1)
        self.assertFalse(self.target.exists())

    def test_byte_limits_content_length_and_http_failures(self):
        responses = [FakeResponse(headers={'Content-Length': '101'}),
                     FakeResponse(data=b'x'*101), FakeResponse(data=b'x', headers={'Content-Length': '2'}),
                     FakeResponse(status=403)]
        for response in responses:
            with self.subTest(response=response.status_code), patch('requests.get', return_value=response):
                with self.assertRaises((r.RecoveryError, OSError)):
                    self.transport.download(self.version, self.remote, self.target, 100)
            self.target.unlink(missing_ok=True)

    def test_pagination_cap_prevents_unbounded_inventory_requests(self):
        self.token = 'same-token'; self.files = []
        with patch('requests.get') as get:
            with self.assertRaises(r.RecoveryError):
                self.transport.download(self.version, self.remote, self.target, 100)
            get.assert_not_called()
        self.assertEqual(len(self.requests), 10)
