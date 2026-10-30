"""Frozen, offline inference assets. No credentials, data or random-weight fallback."""
import hashlib
import json
from pathlib import Path

CONTRACT_SHA256 = 'a80dce8f27f12645539a0312a3ed96af0b5d10b5d7e1cd522af53b9347409c61'
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = ROOT / 'artifacts/inference-v8'
CONTRACT_PATH = ROOT / 'results/reliability-v6/candidate-contract.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verified_contract(path=CONTRACT_PATH):
    if digest(path) != CONTRACT_SHA256:
        raise ValueError('Frozen candidate contract checksum mismatch')
    return json.loads(Path(path).read_text())


def required_files(contract):
    return {
        'best.pt': contract['checkpoint_sha256'],
        'temperature.json': contract['temperature_sha256'],
        'quality-normalizer.npz': contract['quality_normalizer_sha256'],
        'candidate-pipeline.json': contract['artifact_storage']['candidate_sha256'],
        'candidate-contract.json': CONTRACT_SHA256,
    }


def verify_bundle(folder=DEFAULT_BUNDLE):
    folder = Path(folder)
    try:
        contract = verified_contract(folder / 'candidate-contract.json')
        for name, expected in required_files(contract).items():
            if (folder / name).is_symlink() or digest(folder / name) != expected:
                raise ValueError(f'Frozen inference artifact checksum mismatch: {name}')
    except FileNotFoundError as error:
        raise FileNotFoundError('Frozen bundle missing. Run python scripts/install_inference_bundle.py; random weights are never used.') from error
    return contract
