"""Reuse a trusted, complete validation stage only after provenance/raw hash checks."""
import importlib.metadata
import json
from pathlib import Path
import re

from trustretina.config import CLASSES
from .validation import DataValidationError, canonical_hash, file_hash, inside, parse_workbook
from .duplicates import verified_components


def resume_validation(config, stage, *, expected_sha256, expected_nonce,
                      expected_source_commit, expected_source_hashes, expected_packages):
    """Trust anchors must come from a separately verified pinned execution receipt.

    Never obtain expected_sha256 from the same untrusted file being accepted.
    This reuses image decoding only; workbook parsing and raw hashes execute again.
    Split preparation and its support/leakage audits always execute again.
    """
    if (not re.fullmatch('[a-f0-9]{64}', expected_sha256)
            or not re.fullmatch('[a-f0-9]{32}', expected_nonce)
            or not re.fullmatch('[a-f0-9]{40}', expected_source_commit)
            or file_hash(stage) != expected_sha256):
        raise DataValidationError('Validation checkpoint checksum or trust anchor mismatch')
    snapshot = json.loads(Path(stage).read_text())
    if (snapshot.get('stage') != 'validated_inventory'
            or snapshot.get('nonce') != expected_nonce
            or snapshot.get('source_commit') != expected_source_commit
            or snapshot.get('training') is not False or snapshot.get('gpu_requested') is not False):
        raise DataValidationError('Validation checkpoint execution identity/stage mismatch')
    sources = {'validation.py': Path(__file__).with_name('validation.py'),
               'duplicates.py': Path(__file__).with_name('duplicates.py'),
               'config.py': Path(__file__).parents[1] / 'config.py'}
    if (set(expected_source_hashes) != set(sources)
            or any(file_hash(path) != expected_source_hashes[name] for name, path in sources.items())):
        raise DataValidationError('Validation implementation changed; fresh validation required')
    if (set(expected_packages) != {'Pillow', 'openpyxl'}
            or any(importlib.metadata.version(name) != version for name, version in expected_packages.items())):
        raise DataValidationError('Validation dependencies changed; fresh validation required')
    inventory = snapshot['payload']
    if canonical_hash({k:v for k,v in inventory.items() if k != 'fingerprint'}) != inventory['fingerprint']:
        raise DataValidationError('Validation checkpoint inventory fingerprint mismatch')
    if (inventory['class_order'] != CLASSES or config['class_order'] != CLASSES
            or inventory['label_source'] != 'patient' or config['data']['label_source'] != 'patient'
            or inventory['data_kind'] != config['data'].get('kind', 'original_odir')
            or inventory['near_duplicate_hamming'] != config['data'].get('near_duplicate_hamming', 2)
            or inventory['duplicate_policy'] != config['data'].get('duplicate_policy', 'reject')):
        raise DataValidationError('Validation checkpoint does not match data settings')
    root = Path(config['data']['root']).resolve()
    workbook = inside(root, inventory['workbook']['path'])
    if (workbook != inside(root, config['data']['labels'])
            or file_hash(workbook) != inventory['workbook']['sha256']):
        raise DataValidationError('Raw workbook changed since validation')
    parsed, sheet = parse_workbook(workbook, config['data'].get('sheet'), inventory['data_kind'])
    if sheet != inventory['workbook']['sheet'] or len(parsed) != len(inventory['patients']):
        raise DataValidationError('Workbook sheet/selection changed; fresh validation required')
    images = inside(root, config['data']['images'])
    for original, cached in zip(parsed, inventory['patients']):
        if (original['patient_id'] != cached['patient_id'] or original['labels'] != cached['labels']
                or any(inside(images, original['eyes'][eye]) != inside(root, cached['eyes'][eye]['path'])
                       for eye in ('left', 'right'))):
            raise DataValidationError('Workbook labels/eye references changed; fresh validation required')
    for patient in inventory['patients']:
        if set(patient['eyes']) != {'left', 'right'}:
            raise DataValidationError('Cached patient is missing an eye')
        for record in patient['eyes'].values():
            path = inside(root, record['path'])
            if not path.is_relative_to(images) or file_hash(path) != record['sha256']:
                raise DataValidationError('Raw image changed since validation')
    if file_hash(workbook) != inventory['workbook']['sha256']:
        raise DataValidationError('Raw workbook changed since validation')
    verified_components(inventory)
    return inventory, {'checkpoint_sha256': expected_sha256,
                       'source_commit': expected_source_commit, 'source_nonce': expected_nonce,
                       'source_code_sha256': expected_source_hashes, 'packages': expected_packages,
                       'dataset_fingerprint': inventory['fingerprint'], 'raw_hashes_revalidated': True}
