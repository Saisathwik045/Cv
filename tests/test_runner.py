"""Mocked runner transport tests; these do NOT verify a remote Kaggle account."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from trustretina.runner.kaggle import KaggleCLI, RunnerError, check_runner, prepare_smoke, run_smoke, smoke_source


class FakeClient:
    """Controlled responses and downloads, never calls the network."""
    def __init__(self, bundle, failure=None, gpu_ok=True, wrong_nonce=False):
        self.bundle = Path(bundle)
        self.calls = []
        self.failure = failure
        self.gpu_ok = gpu_ok
        self.wrong_nonce = wrong_nonce
        self.polls = 0

    def call(self, *args):
        self.calls.append(args)
        if args[1] == self.failure:
            raise RunnerError('Simulated transport failure')
        if args[1] == 'push':
            return 'Kernel version 7 successfully pushed. Please check progress.'
        if args[1] == 'status':
            self.polls += 1
            return f'owner/kernel/7 has status "{"running" if self.polls == 1 else "complete"}"'
        if args[1] == 'output':
            request = json.loads((self.bundle / 'request.json').read_text())
            data = {'nonce': 'wrong' if self.wrong_nonce else request['nonce'], 'training': False,
                    'gpu_requested': request['gpu'], 'cuda_tensor_ok': self.gpu_ok}
            payload = json.dumps(data).encode()
            download = Path(args[args.index('-p') + 1])
            (download / 'smoke.json').write_bytes(payload)
            (download / 'smoke.sha256').write_text(hashlib.sha256(payload).hexdigest())
        return ''


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name) / 'smoke'

    def run_fake(self, **kwargs):
        prepare_smoke(self.bundle, 'synthetic-test-owner', gpu=True)
        fake = FakeClient(self.bundle, **kwargs)
        return run_smoke(self.bundle, client=fake, sleep=lambda _: None), fake

    def test_prepare_only_never_claims_execution_or_quota(self):
        report = prepare_smoke(self.bundle, 'synthetic-test-owner', gpu=True)
        self.assertFalse(report['smoke_executed'])
        self.assertFalse(report['gpu_quota_used'])
        metadata = json.loads((self.bundle / 'kernel-metadata.json').read_text())
        self.assertTrue(metadata['is_private'])
        self.assertFalse(metadata['enable_internet'])
        self.assertEqual(metadata['dataset_sources'], [])

    def test_all_operations_required_and_version_pinned(self):
        report, fake = self.run_fake()
        for field in ('authentication_verified', 'submission_verified', 'status_polling_verified',
                      'artifact_retrieval_verified', 'unattended_gpu_execution_verified'):
            self.assertTrue(report[field])
        self.assertEqual(fake.polls, 2)
        for args in fake.calls:
            if args[1] in ('status', 'output'):
                self.assertTrue(args[2].endswith('/7'))
        self.assertTrue(run_smoke(self.bundle, client=fake)['unattended_gpu_execution_verified'])
        self.assertEqual(sum(c[1] == 'push' for c in fake.calls), 1)

    def test_each_transport_failure_prevents_gpu_verification(self):
        for operation in ('list', 'push', 'status', 'output'):
            with self.subTest(operation=operation):
                bundle = Path(self.temp.name) / operation
                prepare_smoke(bundle, 'synthetic-test-owner', gpu=True)
                fake = FakeClient(bundle, failure=operation)
                report = run_smoke(bundle, client=fake, sleep=lambda _: None)
                self.assertFalse(report['unattended_gpu_execution_verified'])
                self.assertIn('blocker', report)

    def test_sdk_enum_status_resumes_pinned_job_without_resubmission(self):
        prepare_smoke(self.bundle, 'synthetic-test-owner', gpu=True)
        first = FakeClient(self.bundle, failure='status')
        run_smoke(self.bundle, client=first)
        class EnumStatus(FakeClient):
            def call(self, *args):
                output = super().call(*args)
                if args[1] == 'status':
                    return output.replace('"running"', '"KernelWorkerStatus.RUNNING"').replace(
                        '"complete"', '"KernelWorkerStatus.COMPLETE"')
                return output
        second = EnumStatus(self.bundle)
        report = run_smoke(self.bundle, client=second, sleep=lambda _: None)
        self.assertTrue(report['unattended_gpu_execution_verified'])
        self.assertFalse(any(c[1] == 'push' for c in second.calls))

    def test_stale_artifact_and_missing_cuda_prevent_gpu_verification(self):
        report, _ = self.run_fake(wrong_nonce=True)
        self.assertFalse(report['artifact_retrieval_verified'])
        self.assertFalse(report['unattended_gpu_execution_verified'])
        other = Path(self.temp.name) / 'nogpu'
        prepare_smoke(other, 'synthetic-test-owner', gpu=True)
        report = run_smoke(other, client=FakeClient(other, gpu_ok=False), sleep=lambda _: None)
        self.assertTrue(report['artifact_retrieval_verified'])
        self.assertFalse(report['unattended_gpu_execution_verified'])

    def test_download_checksum_corruption_is_not_verified(self):
        prepare_smoke(self.bundle, 'synthetic-test-owner', gpu=True)
        class Corrupt(FakeClient):
            def call(self, *args):
                output = super().call(*args)
                if args[1] == 'output':
                    (self.bundle / 'download/smoke.sha256').write_text('bad')
                return output
        report = run_smoke(self.bundle, client=Corrupt(self.bundle), sleep=lambda _: None)
        self.assertFalse(report['artifact_retrieval_verified'])
        self.assertFalse(report['unattended_gpu_execution_verified'])

    def test_cpu_smoke_never_verifies_gpu_or_uses_gpu_quota(self):
        prepare_smoke(self.bundle, 'synthetic-test-owner')
        report = run_smoke(self.bundle, client=FakeClient(self.bundle), sleep=lambda _: None)
        self.assertTrue(report['smoke_executed'])
        self.assertFalse(report['unattended_gpu_execution_verified'])
        self.assertFalse(report['gpu_quota_used'])

    def test_edited_script_cannot_launch_training(self):
        prepare_smoke(self.bundle, 'synthetic-test-owner')
        (self.bundle / 'smoke.py').write_text('raise RuntimeError("edited")')
        fake = FakeClient(self.bundle)
        with self.assertRaises(RunnerError):
            run_smoke(self.bundle, client=fake)
        self.assertEqual(fake.calls, [])

    def test_cli_errors_never_echo_credentials(self):
        result = subprocess.CompletedProcess([], 1, stdout='SECRET-VALUE', stderr='SECRET-VALUE')
        with patch('trustretina.runner.kaggle.shutil.which', return_value='kaggle'):
            with patch('trustretina.runner.kaggle.subprocess.run', return_value=result):
                with self.assertRaises(RunnerError) as error:
                    KaggleCLI().call('kernels', 'list', '--mine')
        self.assertNotIn('SECRET-VALUE', str(error.exception))

    def test_absent_credentials_or_cli_are_not_success(self):
        observed = {'kaggle_cli_available': False, 'kaggle_credential_presence_only': {'token_variable': False}}
        with patch('trustretina.runner.kaggle.environment_report', return_value=observed):
            with patch('trustretina.runner.kaggle.probe_url', return_value={'reachable': False}):
                report = check_runner()
        self.assertFalse(report['authentication_verified'])
        self.assertFalse(report['unattended_gpu_execution_verified'])

    def test_poll_timeout_does_not_resubmit(self):
        prepare_smoke(self.bundle, 'synthetic-test-owner', gpu=True)
        class Queued(FakeClient):
            def call(self, *args):
                if args[1] == 'status':
                    return 'job has status "queued"'
                return super().call(*args)
        fake = Queued(self.bundle)
        clock = iter([0, 20]).__next__
        report = run_smoke(self.bundle, timeout_seconds=10, client=fake, clock=clock)
        self.assertIn('deadline', report['blocker'])
        self.assertFalse(report['unattended_gpu_execution_verified'])
        self.assertEqual(report['gpu_quota_used'], 'unknown_after_submission')

    def test_dataset_attached_directly_and_generated_discovery_executes(self):
        # Synthetic mount only: this test says nothing about real Kaggle contents.
        import sys
        from synthetic import make_fixture
        config = make_fixture(self.temp.name)
        reference = 'andrewmvd/ocular-disease-recognition-odir5k'
        mount = Path(self.temp.name) / reference.split('/')[-1]
        Path(config['data']['root']).rename(mount)
        dataset = {'dataset_ref': reference, 'input_root': self.temp.name,
                   'image_directory_name': 'Training Images'}
        prepare_smoke(self.bundle, 'synthetic-test-owner', dataset=dataset)
        metadata = json.loads((self.bundle / 'kernel-metadata.json').read_text())
        self.assertEqual(metadata['dataset_sources'], [reference])
        result = subprocess.run([sys.executable, str(self.bundle / 'smoke.py')],
                                cwd=self.bundle, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.bundle / 'smoke.json').read_text())
        self.assertTrue(report['dataset_attachment_verified'])
        self.assertEqual(report['dataset_ref'], reference)
        self.assertEqual(Path(report['workbooks'][0]['path']).name, 'data.xlsx')
        self.assertTrue(report['image_samples'])
        self.assertFalse(report['cuda_tensor_ok'])

    def test_missing_dataset_artifact_evidence_blocks_gpu_verification(self):
        dataset = {'dataset_ref': 'andrewmvd/ocular-disease-recognition-odir5k',
                   'input_root': '/synthetic/input', 'image_directory_name': 'Training Images'}
        prepare_smoke(self.bundle, 'synthetic-test-owner', gpu=True, dataset=dataset)
        report = run_smoke(self.bundle, client=FakeClient(self.bundle), sleep=lambda _: None)
        self.assertFalse(report['dataset_attachment_verified'])
        self.assertTrue(report['artifact_retrieval_verified'])
        self.assertTrue(report['cuda_tensor_ok'])
        self.assertFalse(report['unattended_gpu_execution_verified'])
        self.assertIn('Dataset attachment', report['blocker'])

    def test_job_version_durable_before_poll_and_resumes_without_push(self):
        prepare_smoke(self.bundle, 'synthetic-test-owner', gpu=True)
        class Interrupted(FakeClient):
            def call(self, *args):
                if args[1] == 'status':
                    receipt = json.loads((self.bundle / 'verification.json').read_text())
                    if not receipt['kernel_version'].endswith('/7'):
                        raise AssertionError('Version not persisted before polling')
                    raise KeyboardInterrupt()
                return super().call(*args)
        first = Interrupted(self.bundle)
        with self.assertRaises(KeyboardInterrupt):
            run_smoke(self.bundle, client=first)
        second = FakeClient(self.bundle)
        report = run_smoke(self.bundle, client=second, sleep=lambda _: None)
        self.assertTrue(report['unattended_gpu_execution_verified'])
        self.assertFalse(any(c[1] == 'push' for c in second.calls))
        self.assertEqual(sum(c[1] == 'push' for c in first.calls), 1)

    def test_ambiguous_submission_never_retries_push(self):
        prepare_smoke(self.bundle, 'synthetic-test-owner', gpu=True)
        failed = FakeClient(self.bundle, failure='push')
        report = run_smoke(self.bundle, client=failed)
        self.assertIn('kernel_id', json.loads((self.bundle / 'verification.json').read_text()))
        second = FakeClient(self.bundle)
        report = run_smoke(self.bundle, client=second)
        self.assertIn('ambiguous', report['blocker'])
        self.assertFalse(any(c[1] == 'push' for c in second.calls))

class MountDiscoveryTests(unittest.TestCase):
    def test_nested_mount_and_independent_cuda_evidence(self):
        from synthetic import make_fixture
        from trustretina.runner.smoke_runtime import run_checks
        import os
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nested = root / 'input' / 'datasets' / 'owner' / 'different-mount' / 'version' / 'ODIR-5K'
            nested.mkdir(parents=True)
            config = make_fixture(nested, count=2)
            dataset = {'dataset_ref': 'andrewmvd/ocular-disease-recognition-odir5k',
                       'input_root': str(root / 'input'), 'image_directory_name': 'Training Images'}
            previous = Path.cwd()
            os.chdir(root)
            try:
                report = run_checks('test', True, dataset, cuda_check=lambda: {'cuda_tensor_ok': True})
                self.assertTrue(report['checks_passed'])
                self.assertIn('datasets/owner/different-mount/version/ODIR-5K', report['workbooks'][0]['path'])
                Path(config['data']['root'], 'data.xlsx').unlink()
                report = run_checks('test', True, dataset, cuda_check=lambda: {'cuda_tensor_ok': True})
                self.assertTrue(report['cuda_tensor_ok'])
                self.assertFalse(report['checks_passed'])
                self.assertIn('No original source', report['dataset_error'])
                self.assertEqual(json.loads(Path('smoke.json').read_text()), report)
                self.assertEqual(hashlib.sha256(Path('smoke.json').read_bytes()).hexdigest(), Path('smoke.sha256').read_text())
            finally:
                os.chdir(previous)

    def test_ambiguous_original_sources_fail_closed(self):
        from synthetic import make_fixture
        from trustretina.runner.smoke_runtime import discover_source
        with tempfile.TemporaryDirectory() as temp:
            for name in ('one', 'two'):
                parent = Path(temp, name)
                parent.mkdir()
                make_fixture(parent, count=1)
            with self.assertRaisesRegex(RuntimeError, 'Ambiguous original sources'):
                discover_source({'input_root': temp, 'image_directory_name': 'Training Images', 'dataset_ref': 'owner/source'})

    def test_new_version_preserves_previous_receipt_and_kernel_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            old, new = Path(temp, 'old'), Path(temp, 'new')
            prepare_smoke(old, 'owner', gpu=True)
            run_smoke(old, client=FakeClient(old, failure='status'))
            before = (old / 'verification.json').read_bytes()
            prepare_smoke(new, 'owner', gpu=True, previous_bundle=old)
            self.assertEqual((old / 'verification.json').read_bytes(), before)
            self.assertEqual(json.loads((new / 'kernel-metadata.json').read_text())['id'],
                             json.loads((old / 'kernel-metadata.json').read_text())['id'])
            self.assertNotEqual(json.loads((new / 'request.json').read_text())['nonce'],
                                json.loads((old / 'request.json').read_text())['nonce'])
