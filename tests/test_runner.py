"""Mocked runner transport tests; these do NOT verify a remote Kaggle account."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from trustretina.runner.kaggle import KaggleCLI, RunnerError, check_runner, prepare_smoke, run_smoke


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
        with self.assertRaises(FileExistsError):
            run_smoke(self.bundle, client=fake)
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
