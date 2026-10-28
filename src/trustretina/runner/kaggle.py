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


def smoke_source(nonce, gpu):
    return f'''# TrustRetina nontraining smoke test. No data downloads or model training.
import hashlib
import json
from pathlib import Path
report = {{"nonce": {nonce!r}, "training": False, "gpu_requested": {gpu!r}, "cuda_tensor_ok": False}}
if {gpu!r}:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    result = (torch.ones(1, device="cuda") + 1).item()
    torch.cuda.synchronize()
    if result != 2:
        raise RuntimeError("CUDA arithmetic failed")
    report.update(cuda_tensor_ok=True, torch_version=torch.__version__, device=torch.cuda.get_device_name(0))
else:
    if 1 + 1 != 2:
        raise RuntimeError("CPU arithmetic failed")
payload = json.dumps(report, sort_keys=True).encode()
Path("smoke.json").write_bytes(payload)
Path("smoke.sha256").write_text(hashlib.sha256(payload).hexdigest())
'''


def smoke_metadata(owner, nonce, gpu):
    slug = f'trustretina-smoke-{nonce}'
    return {'id': f'{owner}/{slug}', 'title': slug, 'code_file': 'smoke.py',
            'language': 'python', 'kernel_type': 'script', 'is_private': True,
            'enable_gpu': gpu, 'enable_internet': False,
            'dataset_sources': [], 'competition_sources': [], 'kernel_sources': []}


def prepare_smoke(destination, owner, gpu=False):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', owner):
        raise RunnerError('Invalid Kaggle owner slug')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    nonce = uuid.uuid4().hex
    metadata = smoke_metadata(owner, nonce, gpu)
    (destination / 'smoke.py').write_text(smoke_source(nonce, gpu))
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    (destination / 'request.json').write_text(json.dumps({'owner': owner, 'nonce': nonce, 'gpu': gpu}))
    return {'prepared': True, 'smoke_executed': False, 'gpu_quota_used': False,
            'gpu_requested': gpu, 'bundle': str(destination)}


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
    expected = smoke_metadata(owner, nonce, gpu)
    if ((bundle / 'smoke.py').read_text() != smoke_source(nonce, gpu)
            or json.loads((bundle / 'kernel-metadata.json').read_text()) != expected):
        raise RunnerError('Bundle differs from the fixed nontraining smoke template')
    report = {'authentication_verified': False, 'submission_verified': False,
              'status_polling_verified': False, 'artifact_retrieval_verified': False,
              'unattended_gpu_execution_verified': False, 'gpu_requested': gpu,
              'smoke_executed': False, 'gpu_quota_used': False}
    # Exclusive receipt remains after ambiguous/failed submission; rerunning must not
    # accidentally consume quota twice. Poll/download an existing version manually.
    receipt = bundle / 'verification.json'
    with receipt.open('x') as handle:
        json.dump(report, handle)
    try:
        client.call('kernels', 'list', '--mine')
        report['authentication_verified'] = True
        report['gpu_quota_used'] = 'unknown_after_submission' if gpu else False
        output = client.call('kernels', 'push', '-p', str(bundle), '--timeout', '120')
        match = re.search(r'Kernel version (\d+) successfully pushed', output, re.I)
        if not match:
            raise RunnerError('Submission version not verified from CLI response; do not resubmit automatically')
        version = f"{expected['id']}/{int(match.group(1))}"
        report.update(submission_verified=True, kernel_version=version)
        deadline = clock() + timeout_seconds
        while True:
            output = client.call('kernels', 'status', version)
            status = re.search(r'has status "([A-Za-z_]+)"', output)
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
        download.mkdir(exist_ok=False)
        client.call('kernels', 'output', version, '-p', str(download), '--file-pattern', r'^smoke\.(json|sha256)$')
        for name in ('smoke.json', 'smoke.sha256'):
            if (download / name).is_symlink() or not (download / name).is_file():
                raise RunnerError('Expected smoke artifact missing or unsafe')
        payload = (download / 'smoke.json').read_bytes()
        if len(payload) > 16384 or hashlib.sha256(payload).hexdigest() != (download / 'smoke.sha256').read_text().strip():
            raise RunnerError('Smoke artifact checksum mismatch')
        remote = json.loads(payload)
        if remote.get('nonce') != nonce or remote.get('training') is not False or remote.get('gpu_requested') is not gpu:
            raise RunnerError('Smoke artifact identity mismatch')
        report.update(artifact_retrieval_verified=True, smoke_executed=True,
                      gpu_quota_used=gpu, remote_report=remote)
        report['unattended_gpu_execution_verified'] = gpu and remote.get('cuda_tensor_ok') is True
        if gpu and not report['unattended_gpu_execution_verified']:
            raise RunnerError('Artifacts retrieved but CUDA arithmetic not verified')
    except (RunnerError, OSError, ValueError, KeyError, TypeError) as exc:
        report['blocker'] = str(exc) if isinstance(exc, RunnerError) else 'Local artifact or receipt validation failed'
    finally:
        # Reports contain only structured evidence; never raw CLI stdout/stderr.
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(receipt)
    return report
