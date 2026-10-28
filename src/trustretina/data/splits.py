"""Deterministic patient-level iterative stratification and immutable manifests."""
import json
import importlib.metadata
import platform
import subprocess
import os
from pathlib import Path
import random
import shutil
import tempfile

from trustretina.config import CLASSES, PARTITIONS
from .validation import DataValidationError, canonical_hash, file_hash, validate_dataset


def make_splits(inventory, config):
    patients = inventory['patients']
    settings = config['split']
    n = len(patients)
    minimum_pos = settings['minimum_positive_patients']
    minimum_neg = settings['minimum_negative_patients']
    totals = [sum(p['labels'][j] for p in patients) for j in range(8)]
    if any(t < 4 * minimum_pos or n - t < 4 * minimum_neg for t in totals):
        raise DataValidationError('Insufficient global per-class support for four partitions')
    fractions = settings['fractions']
    sizes = [int(n * f) for f in fractions]
    order = sorted(range(4), key=lambda k: (-(n * fractions[k] - sizes[k]), k))
    for k in order[:n - sum(sizes)]:
        sizes[k] += 1
    if any(s < minimum_pos + minimum_neg for s in sizes):
        raise DataValidationError('Partition capacity cannot meet required positive/negative support')
    # Allocate the rarest remaining label first, choosing its greatest unfilled
    # per-partition target. No seed retries or final-evaluation outcome selection.
    remaining = set(range(n))
    capacity = sizes.copy()
    need = [[total * f for total in totals] for f in fractions]
    groups = [[] for _ in range(4)]
    rng = random.Random(config['seed'])
    ties = list(range(n))
    rng.shuffle(ties)
    rank = {index: position for position, index in enumerate(ties)}
    while remaining:
        counts = [sum(patients[i]['labels'][j] for i in remaining) for j in range(8)]
        nonzero = [j for j in range(8) if counts[j]]
        label = min(nonzero, key=lambda j: (counts[j], j)) if nonzero else None
        selected = sorted((i for i in remaining if label is None or patients[i]['labels'][label]),
                          key=lambda i: rank[i])
        for i in selected:
            available = [k for k in range(4) if capacity[k] > 0]
            best = max(available, key=lambda k: (need[k][label] if label is not None else 0,
                                                 capacity[k], -k))
            groups[best].append(patients[i])
            capacity[best] -= 1
            for j, value in enumerate(patients[i]['labels']):
                need[best][j] -= value
            remaining.remove(i)
    manifest = {'schema_version': 1, 'algorithm': 'trustretina-iterative-rarest-label-v1',
                'dataset_fingerprint': inventory['fingerprint'], 'class_order': CLASSES,
                'label_source': 'patient', 'seed': config['seed'], 'split_settings': settings,
                'partitions': {name: sorted(groups[k], key=lambda p: p['patient_id'])
                               for k, name in enumerate(PARTITIONS)}}
    manifest['support'] = audit_manifest(manifest, inventory)
    return manifest


def audit_manifest(manifest, inventory):
    if (manifest['class_order'] != CLASSES or manifest['label_source'] != 'patient'
            or set(manifest['partitions']) != set(PARTITIONS)
            or manifest['dataset_fingerprint'] != inventory['fingerprint']):
        raise DataValidationError('Manifest schema/partition/dataset mismatch')
    expected = {p['patient_id']: p for p in inventory['patients']}
    seen, hashes = set(), set()
    support = {}
    for partition in PARTITIONS:
        rows = manifest['partitions'][partition]
        for row in rows:
            pid = row['patient_id']
            if pid in seen or expected.get(pid) != row or set(row['eyes']) != {'left', 'right'}:
                raise DataValidationError('Patient leakage, altered labels/eyes or unknown patient')
            seen.add(pid)
            for eye in row['eyes'].values():
                for kind in ['sha256', 'pixel_sha256']:
                    key = (kind, eye[kind])
                    if key in hashes:
                        raise DataValidationError('Duplicate content in manifest')
                    hashes.add(key)
        positive = [sum(p['labels'][j] for p in rows) for j in range(8)]
        negative = [len(rows) - count for count in positive]
        settings = manifest['split_settings']
        if (min(positive) < settings['minimum_positive_patients']
                or min(negative) < settings['minimum_negative_patients']):
            raise DataValidationError(f'Insufficient per-class support in {partition}; no manifest published')
        support[partition] = {'patients': len(rows), 'images': 2 * len(rows),
                              'positive_patients': dict(zip(CLASSES, positive)),
                              'negative_patients': dict(zip(CLASSES, negative))}
    if seen != set(expected):
        raise DataValidationError('Manifest missing patients')
    return support


def implementation_provenance():
    implementation = canonical_hash({name: file_hash(Path(__file__).with_name(name))
                                     for name in ('validation.py', 'splits.py')})
    try:
        commit = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[3]),
                                          'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
                                         timeout=5).decode().strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {'implementation_sha256': implementation, 'source_commit': commit,
            'python': platform.python_version(),
            'packages': {name: importlib.metadata.version(name) for name in ('openpyxl', 'Pillow')}}


def prepare_data(config, destination):
    """Validate every input on every call; publish a bundle only after all checks pass."""
    destination = Path(destination).resolve()
    root = Path(config['data']['root']).resolve()
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise DataValidationError('Artifact bundle and raw dataset must have disjoint paths')
    inventory = validate_dataset(config)
    provenance = implementation_provenance()
    key = canonical_hash({'inventory': inventory['fingerprint'], 'seed': config['seed'],
                          'split': config['split'], 'implementation': provenance['implementation_sha256'], 'algorithm': 'trustretina-iterative-rarest-label-v1'})
    if destination.exists():
        completion = json.loads((destination / 'complete.json').read_text())
        if completion['input_key'] != key:
            raise DataValidationError('Existing bundle has different inputs/settings; refusing to overwrite')
        expected_files = {'inventory.json', 'manifest.json', 'provenance.json', *[f'{p}.json' for p in PARTITIONS]}
        if set(completion['files']) != expected_files:
            raise DataValidationError('Incomplete bundle marker')
        for name, digest in completion['files'].items():
            if file_hash(destination / name) != digest:
                raise DataValidationError('Bundle checksum mismatch; refusing reuse')
        saved_inventory = json.loads((destination / 'inventory.json').read_text())
        saved = json.loads((destination / 'manifest.json').read_text())
        if saved_inventory != inventory or saved['split_settings'] != config['split'] or saved['seed'] != config['seed']:
            raise DataValidationError('Bundle provenance mismatch')
        support = audit_manifest(saved, inventory)
        if support != saved['support']:
            raise DataValidationError('Bundle support mismatch')
        for partition in PARTITIONS:
            if json.loads((destination / f'{partition}.json').read_text()) != saved['partitions'][partition]:
                raise DataValidationError('Partition file mismatch')
        return {'reused': True, 'data_kind': inventory['data_kind'], 'dataset_fingerprint': inventory['fingerprint'],
                'bundle': str(destination), 'support': support}
    manifest = make_splits(inventory, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive lock + directory rename prevents partial or competing publication.
    lock = destination.with_name(destination.name + '.lock')
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    staging = None
    try:
        staging = Path(tempfile.mkdtemp(prefix='.prepare-', dir=destination.parent))
        records = {'inventory.json': inventory, 'manifest.json': manifest, 'provenance.json': provenance,
                   **{f'{p}.json': manifest['partitions'][p] for p in PARTITIONS}}
        checksums = {}
        for name, value in records.items():
            path = staging / name
            path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
            checksums[name] = file_hash(path)
        (staging / 'complete.json').write_text(json.dumps({'schema_version': 1, 'input_key': key,
                                                          'files': checksums}, indent=2) + '\n')
        if destination.exists():
            raise DataValidationError('Another process published the bundle; rerun to verify it')
        staging.rename(destination)
    finally:
        os.close(descriptor)
        lock.unlink()
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    return {'reused': False, 'data_kind': inventory['data_kind'], 'dataset_fingerprint': inventory['fingerprint'],
            'bundle': str(destination), 'support': manifest['support']}
