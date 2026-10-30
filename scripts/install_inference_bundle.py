#!/usr/bin/env python3
"""Restore only pinned inference assets. Kaggle credentials needed only for --download."""
import argparse
import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trustretina.inference.bundle import DEFAULT_BUNDLE, CONTRACT_PATH, verified_contract, required_files, digest, verify_bundle


def install(destination, *, checkpoint=None, parameters=None, download=False):
    contract = verified_contract()
    destination = Path(destination)
    if destination.exists():
        verify_bundle(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.bundle-') as tmp:
        tmp = Path(tmp)
        if download:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    from kaggle import api
                except ImportError as error:
                    raise RuntimeError('Private recovery requires the optional SDK: python -m pip install kaggle==2.2.5; use the existing authorized Kaggle account. App runtime does not need it.') from error
                for version, pattern, folder in (
                    (contract['selected_checkpoint_kernel_version'], r'^full/best\.pt$', tmp / 'weights'),
                    (contract['artifact_storage']['analysis_kernel_version'], r'^full/(temperature\.json|quality-normalizer\.npz|candidate-pipeline\.json)$', tmp / 'parameters'),
                ):
                    token = None
                    while True:
                        _, token = api.kernels_output(version, path=str(folder), file_pattern=pattern, page_token=token, page_size=100)
                        if not token:
                            break
            checkpoint = tmp / 'weights/full/best.pt'
            parameters = tmp / 'parameters/full'
        if not checkpoint or not parameters:
            raise ValueError('Supply --download or both --checkpoint and --parameters; no training or dataset required')
        staged = tmp / 'bundle'
        staged.mkdir()
        for name, expected in required_files(contract).items():
            source = CONTRACT_PATH if name == 'candidate-contract.json' else Path(checkpoint) if name == 'best.pt' else Path(parameters) / name
            if digest(source) != expected:
                raise ValueError(f'Pinned artifact mismatch: {name}')
            shutil.copyfile(source, staged / name)
        verify_bundle(staged)
        (staged / 'recovery.json').write_text(json.dumps({'checkpoint_version': contract['selected_checkpoint_kernel_version'], 'parameters_version': contract['artifact_storage']['analysis_kernel_version'], 'hashes': required_files(contract)}, indent=2))
        staged.rename(destination)
    return destination


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, default=DEFAULT_BUNDLE)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--parameters', type=Path)
    p.add_argument('--download', action='store_true')
    args = p.parse_args()
    try:
        print('Verified offline bundle:', install(args.destination, checkpoint=args.checkpoint, parameters=args.parameters, download=args.download))
    except Exception as error:
        p.exit(1, f'Artifact installation failed: {error}\n')
