"""Standalone nontraining checks embedded in the submitted smoke script."""
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile


def discover_source(dataset):
    """Inspect the configured input tree without assuming a mount name or depth."""
    from PIL import Image
    root = Path(dataset['input_root']).resolve(strict=True)
    workbooks, folders = [], []
    def inaccessible(error):
        raise RuntimeError('Attached inputs are inaccessible') from error
    for current, dirs, files in os.walk(root, followlinks=False, onerror=inaccessible):
        base = Path(current)
        if base.name == dataset['image_directory_name']:
            folders.append(base)
        if 'data.xlsx' in files:
            workbooks.append(base / 'data.xlsx')
    if not workbooks or not folders:
        raise RuntimeError('No original source beneath input root: require data.xlsx and Training Images; CSV is not accepted')
    if len(workbooks) != 1 or len(folders) != 1:
        raise RuntimeError(f'Ambiguous original sources: {len(workbooks)} data.xlsx files and {len(folders)} Training Images directories')
    workbook, folder = workbooks[0], folders[0]
    if not workbook.resolve().is_relative_to(root) or not folder.resolve().is_relative_to(root):
        raise RuntimeError('Original source escapes the configured input root')
    with zipfile.ZipFile(workbook) as archive:
        if 'xl/workbook.xml' not in archive.namelist():
            raise RuntimeError('Invalid XLSX workbook')
    samples = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and re.fullmatch(r'[0-9]+_(left|right)\.(jpg|jpeg|png)', path.name, re.I):
            if not path.resolve().is_relative_to(root):
                raise RuntimeError('Eye image escapes the configured input root')
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                samples.append({'path': str(path.relative_to(root)), 'width': image.width, 'height': image.height})
            if len(samples) == 2:
                break
    if not samples:
        raise RuntimeError('Original-style eye images were not found/decoded')
    return {'dataset_ref': dataset['dataset_ref'], 'dataset_attachment_verified': True,
            'workbooks': [{'path': str(workbook.relative_to(root)),
                           'sha256': hashlib.sha256(workbook.read_bytes()).hexdigest()}],
            'image_directory': str(folder.relative_to(root)), 'image_samples': samples,
            'content_check': 'unique original-style layout and decoded samples; full validation pending'}


def check_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable')
    value = (torch.ones(1, device='cuda') + 1).item()
    torch.cuda.synchronize()
    if value != 2:
        raise RuntimeError('CUDA arithmetic failed')
    return {'cuda_tensor_ok': True, 'torch_version': torch.__version__,
            'device': torch.cuda.get_device_name(0)}


def write_report(report):
    payload = json.dumps(report, sort_keys=True).encode()
    Path('smoke.json').write_bytes(payload)
    Path('smoke.sha256').write_text(hashlib.sha256(payload).hexdigest())


def run_checks(nonce, gpu, dataset, cuda_check=check_cuda):
    report = {'nonce': nonce, 'training': False, 'gpu_requested': gpu,
              'cuda_tensor_ok': False, 'dataset_attachment_verified': False}
    if gpu:
        try:
            report.update(cuda_check())
        except Exception as error:
            report['cuda_error'] = f'{type(error).__name__}: {error}'
    write_report(report)  # CUDA evidence survives later discovery failure.
    if dataset is not None:
        try:
            root = Path(dataset['input_root'])
            report['input_entries'] = sorted(p.name for p in root.iterdir())[:50]
            report.update(discover_source(dataset))
        except Exception as error:
            report['dataset_error'] = f'{type(error).__name__}: {error}'
    report['checks_passed'] = (not gpu or report['cuda_tensor_ok']) and (
        dataset is None or report['dataset_attachment_verified'])
    write_report(report)
    return report
