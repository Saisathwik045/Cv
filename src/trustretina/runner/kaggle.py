"""Run bounded CLI operations without echoing credential-bearing client output."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

from trustretina.environment import environment_report, probe_url


class RunnerError(ValueError):
    """Runner operation was not verified; never includes raw CLI output."""


class KaggleCLI:
    def call(self, *args):
        executable = shutil.which('kaggle')
        if not executable:
            raise RunnerError('Kaggle CLI is not installed')
        try:
            result = subprocess.run([executable, *args], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            raise RunnerError('Kaggle CLI connection/process timeout or startup failure') from None
        if result.returncode:
            raise RunnerError(f'Kaggle CLI failed (exit {result.returncode}); output withheld to protect credentials')
        # Some client commands print an API error but return zero.
        if re.search(r'kernel push error|\bunauthorized\b|\bforbidden\b', result.stdout, re.I):
            raise RunnerError('Kaggle API rejected the operation')
        return result.stdout


def check_runner(client=None):
    client = client or KaggleCLI()
    observations = environment_report()
    credentials = observations['kaggle_credential_presence_only']
    try:
        version = importlib.metadata.version('kaggle')
    except importlib.metadata.PackageNotFoundError:
        version = None
    report = {'client_version': version, 'cli_available': observations['kaggle_cli_available'],
              'credential_presence_only': credentials,
              'kaggle_network_probe': probe_url('https://www.kaggle.com/api/v1/kernels/list'),
              'authentication_verified': False, 'submission_verified': False,
              'status_polling_verified': False, 'artifact_retrieval_verified': False,
              'unattended_gpu_execution_verified': False, 'smoke_executed': False,
              'gpu_quota_used': False}
    if not report['cli_available']:
        report['blocker'] = 'Install the runner extra after managed network access is repaired'
    elif not any(credentials.values()):
        report['blocker'] = 'No supported credential presence observed; bind KAGGLE_API_TOKEN securely'
    else:
        try:
            client.call('kernels', 'list', '--mine')
            report['authentication_verified'] = True
        except RunnerError as exc:
            report['blocker'] = str(exc)
    return report


def smoke_source(nonce, gpu, dataset=None):
    runtime = Path(__file__).with_name('smoke_runtime.py').read_text()
    return runtime + f"\nrun_checks({nonce!r}, {gpu!r}, {dataset!r})\n"


def smoke_metadata(owner, nonce, gpu, dataset=None, previous_version=None):
    slug = previous_version.split('/')[1] if previous_version else f'trustretina-smoke-{nonce}'
    return {'id': f'{owner}/{slug}', 'title': slug, 'code_file': 'smoke.py',
            'language': 'python', 'kernel_type': 'script', 'is_private': True,
            'enable_gpu': gpu, 'enable_internet': False,
            'dataset_sources': [dataset['dataset_ref']] if dataset else [], 'competition_sources': [], 'kernel_sources': []}


def validate_smoke_dataset(dataset):
    if dataset is None:
        return
    if (set(dataset) != {'dataset_ref', 'input_root', 'image_directory_name'}
            or not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+', dataset['dataset_ref'])
            or not Path(dataset['input_root']).is_absolute()
            or Path(dataset['image_directory_name']).name != dataset['image_directory_name']):
        raise RunnerError('Invalid smoke dataset configuration')


def prepare_smoke(destination, owner, gpu=False, dataset=None, previous_bundle=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', owner):
        raise RunnerError('Invalid Kaggle owner slug')
    validate_smoke_dataset(dataset)
    previous_version = None
    if previous_bundle is not None:
        old = Path(previous_bundle).resolve()
        saved = json.loads((old / 'verification.json').read_text())
        previous_version = saved.get('kernel_version', '')
        old_request = json.loads((old / 'request.json').read_text())
        if (not saved.get('submission_verified')
                or not re.fullmatch(re.escape(owner) + r'/[A-Za-z0-9_-]+/[0-9]+', previous_version)
                or saved.get('kernel_id') != previous_version.rsplit('/', 1)[0]
                or saved.get('request_sha256') != hashlib.sha256(json.dumps(old_request, sort_keys=True).encode()).hexdigest()):
            raise RunnerError('Previous pinned submission identity is not verified')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    nonce = uuid.uuid4().hex
    metadata = smoke_metadata(owner, nonce, gpu, dataset, previous_version)
    (destination / 'smoke.py').write_text(smoke_source(nonce, gpu, dataset))
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    (destination / 'request.json').write_text(json.dumps({'owner': owner, 'nonce': nonce, 'gpu': gpu, 'dataset': dataset,
                                                      'previous_kernel_version': previous_version}))
    return {'prepared': True, 'smoke_executed': False, 'gpu_quota_used': False,
            'gpu_requested': gpu, 'dataset_ref': dataset['dataset_ref'] if dataset else None, 'bundle': str(destination)}


def run_smoke(bundle, timeout_seconds=600, client=None, sleep=time.sleep, clock=time.monotonic):
    """Explicit submission; never called by preflight or CI. No automatic re-submission."""
    if not 10 <= timeout_seconds <= 1800:
        raise RunnerError('Polling timeout must be between 10 and 1800 seconds')
    client = client or KaggleCLI()
    bundle = Path(bundle).resolve()
    request = json.loads((bundle / 'request.json').read_text())
    owner, nonce, gpu = request['owner'], request['nonce'], request['gpu']
    if (not re.fullmatch(r'[a-f0-9]{32}', nonce) or type(gpu) is not bool
            or not re.fullmatch(r'[A-Za-z0-9_-]+', owner)):
        raise RunnerError('Invalid smoke request')
    dataset = request.get('dataset')
    validate_smoke_dataset(dataset)
    previous_version = request.get('previous_kernel_version')
    if previous_version is not None and not re.fullmatch(re.escape(owner) + r'/[A-Za-z0-9_-]+/[0-9]+', previous_version):
        raise RunnerError('Invalid previous kernel version')
    expected = smoke_metadata(owner, nonce, gpu, dataset, previous_version)
    if ((bundle / 'smoke.py').read_text() != smoke_source(nonce, gpu, dataset)
            or json.loads((bundle / 'kernel-metadata.json').read_text()) != expected):
        raise RunnerError('Bundle differs from the fixed nontraining smoke template')
    report = {'authentication_verified': False, 'submission_verified': False,
              'status_polling_verified': False, 'artifact_retrieval_verified': False,
              'unattended_gpu_execution_verified': False, 'gpu_requested': gpu,
              'smoke_executed': False, 'gpu_quota_used': False, 'dataset_attachment_verified': False,
              'kernel_id': expected['id'], 'request_sha256': hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()}
    receipt = bundle / 'verification.json'
    def persist():
        temporary = receipt.with_suffix('.tmp')
        with temporary.open('w') as handle:
            json.dump(report, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(receipt)
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved.get('request_sha256') != report['request_sha256'] or saved.get('kernel_id') != expected['id']:
            raise RunnerError('Receipt belongs to different inputs or an older template; refusing submission')
        report = saved
        report.update(unattended_gpu_execution_verified=False, artifact_retrieval_verified=False,
                      smoke_executed=False, dataset_attachment_verified=False)
    else:
        with receipt.open('x') as handle:
            json.dump(report, handle)
    # A concurrent verifier must never submit a second job. A stale lock requires
    # confirming the previous local process is gone, then removing only the lock.
    lock = bundle / 'verification.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        latest = json.loads(receipt.read_text())
        expected_key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if latest.get('request_sha256') != expected_key or latest.get('kernel_id') != expected['id']:
            raise RunnerError('Receipt identity changed; refusing submission')
        report = latest
        report.update(unattended_gpu_execution_verified=False, artifact_retrieval_verified=False,
                      smoke_executed=False, dataset_attachment_verified=False)
        report.pop('blocker', None)
        client.call('kernels', 'list', '--mine')
        report['authentication_verified'] = True
        if report.get('submission_verified'):
            version = report.get('kernel_version', '')
            if not re.fullmatch(re.escape(expected['id']) + r'/[0-9]+', version):
                raise RunnerError('Saved kernel version does not match the prepared job')
        else:
            if report.get('submission_attempted'):
                raise RunnerError('Submission was interrupted/ambiguous; inspect saved kernel_id, never blindly resubmit')
            report.update(submission_attempted=True, gpu_quota_used='unknown_after_submission' if gpu else False)
            persist()  # Kernel ID and intent must survive a process interruption.
            output = client.call('kernels', 'push', '-p', str(bundle), '--timeout', '120')
            match = re.search(r'Kernel version (\d+) successfully pushed', output, re.I)
            if not match:
                raise RunnerError('Submission version not verified from CLI response; do not resubmit automatically')
            version = f"{expected['id']}/{int(match.group(1))}"
            report.update(submission_verified=True, kernel_version=version)
            if previous_version and int(match.group(1)) <= int(previous_version.rsplit('/', 1)[1]):
                persist()
                raise RunnerError('Returned version did not advance; inspect remote job, never resubmit')
            persist()  # Persist the pinned version BEFORE the first status request.
        deadline = clock() + timeout_seconds
        while True:
            output = client.call('kernels', 'status', version)
            status = re.search(r'has status "(?:KernelWorkerStatus\.)?([A-Za-z_]+)"', output)
            if not status:
                raise RunnerError('Unknown status response; execution not verified')
            report['status_polling_verified'] = True
            state = status.group(1).lower()
            if state == 'complete':
                break
            if state not in ('queued', 'running'):
                raise RunnerError('Remote smoke ended without success')
            if clock() >= deadline:
                raise RunnerError('Smoke polling deadline exceeded; remote job may still run, do not resubmit')
            sleep(min(15, max(0, deadline - clock())))
        download = bundle / 'download'
        download.mkdir(exist_ok=True)
        client.call('kernels', 'output', version, '-p', str(download), '--file-pattern', r'^smoke\.(json|sha256)$')
        for name in ('smoke.json', 'smoke.sha256'):
            if (download / name).is_symlink() or not (download / name).is_file():
                raise RunnerError('Expected smoke artifact missing or unsafe')
        payload = (download / 'smoke.json').read_bytes()
        if len(payload) > 16384 or hashlib.sha256(payload).hexdigest() != (download / 'smoke.sha256').read_text().strip():
            raise RunnerError('Smoke artifact checksum mismatch')
        remote = json.loads(payload)
        if not isinstance(remote, dict):
            raise RunnerError('Smoke artifact is not a JSON object')
        if remote.get('nonce') != nonce or remote.get('training') is not False or remote.get('gpu_requested') is not gpu:
            raise RunnerError('Smoke artifact identity mismatch')
        report.update(artifact_retrieval_verified=True, remote_report=remote,
                      gpu_quota_used=gpu, cuda_tensor_ok=remote.get('cuda_tensor_ok') is True)
        if dataset is not None:
            workbooks = remote.get('workbooks', [])
            samples = remote.get('image_samples', [])
            if (remote.get('dataset_ref') != dataset['dataset_ref']
                    or remote.get('dataset_attachment_verified') is not True
                    or not isinstance(workbooks, list) or not workbooks
                    or not isinstance(samples, list) or not samples
                    or any(not isinstance(w, dict) or Path(w.get('path', '')).name != 'data.xlsx'
                           or not re.fullmatch('[a-f0-9]{64}', w.get('sha256', '')) for w in workbooks)):
                raise RunnerError('Dataset attachment/workbook discovery not verified')
            report['dataset_attachment_verified'] = True
        report.update(artifact_retrieval_verified=True, smoke_executed=True,
                      gpu_quota_used=gpu, remote_report=remote)
        report['unattended_gpu_execution_verified'] = gpu and remote.get('cuda_tensor_ok') is True
        if gpu and not report['unattended_gpu_execution_verified']:
            raise RunnerError('Artifacts retrieved but CUDA arithmetic not verified')
    except (RunnerError, OSError, ValueError, KeyError, TypeError) as exc:
        report['blocker'] = str(exc) if isinstance(exc, RunnerError) else 'Local artifact or receipt validation failed'
    finally:
        # Reports contain only structured evidence; never raw CLI stdout/stderr.
        persist()
        os.close(descriptor)
        lock.unlink()
    return report
