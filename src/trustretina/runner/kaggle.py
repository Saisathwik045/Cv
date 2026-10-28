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
    dataset_code = ''
    if dataset is not None:
        dataset_code = f'''
# Input location comes from the submitted configuration, not a Python mount constant.
import zipfile
import re
from PIL import Image
input_root = Path({dataset['input_root']!r})
source_root = input_root / {dataset['dataset_ref'].split('/')[-1]!r}
if not source_root.is_dir():
    raise RuntimeError("Attached dataset mount is missing")
workbooks = sorted(source_root.rglob("data.xlsx"))
image_dirs = sorted(p for p in source_root.rglob({dataset['image_directory_name']!r}) if p.is_dir())
if not workbooks or not image_dirs:
    raise RuntimeError("Original-style data.xlsx/image directory layout not found")
samples = []
for folder in image_dirs:
    for image_path in sorted(folder.iterdir()):
        if image_path.is_file() and re.fullmatch(r"[0-9]+_(left|right)\\.(jpg|jpeg|png)", image_path.name, re.I):
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                samples.append({{"path": str(image_path.relative_to(input_root)), "width": image.width, "height": image.height}})
            if len(samples) == 2:
                break
    if len(samples) == 2:
        break
if not samples:
    raise RuntimeError("Original-style eye images were not found/decoded")
workbook_records = []
for workbook in workbooks:
    with zipfile.ZipFile(workbook) as archive:
        if "xl/workbook.xml" not in archive.namelist():
            raise RuntimeError("Invalid XLSX workbook")
    workbook_records.append({{"path": str(workbook.relative_to(input_root)), "sha256": hashlib.sha256(workbook.read_bytes()).hexdigest()}})
report.update(dataset_ref={dataset['dataset_ref']!r}, dataset_attachment_verified=True,
              workbooks=workbook_records, image_samples=samples,
              content_check="original-style layout and decoded samples; full validation pending")
'''
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
{dataset_code}
payload = json.dumps(report, sort_keys=True).encode()
Path("smoke.json").write_bytes(payload)
Path("smoke.sha256").write_text(hashlib.sha256(payload).hexdigest())
'''


def smoke_metadata(owner, nonce, gpu, dataset=None):
    slug = f'trustretina-smoke-{nonce}'
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


def prepare_smoke(destination, owner, gpu=False, dataset=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', owner):
        raise RunnerError('Invalid Kaggle owner slug')
    validate_smoke_dataset(dataset)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    nonce = uuid.uuid4().hex
    metadata = smoke_metadata(owner, nonce, gpu, dataset)
    (destination / 'smoke.py').write_text(smoke_source(nonce, gpu, dataset))
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    (destination / 'request.json').write_text(json.dumps({'owner': owner, 'nonce': nonce, 'gpu': gpu, 'dataset': dataset}))
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
    expected = smoke_metadata(owner, nonce, gpu, dataset)
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
            persist()  # Persist the pinned version BEFORE the first status request.
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
