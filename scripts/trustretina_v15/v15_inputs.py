"""V14 byte verification and sample alignment. Never deserializes checkpoints."""
import hashlib
import json
from pathlib import Path

import numpy as np

from trustretina.config import CLASSES
from trustretina.evaluation.metrics import aggregate_patient_mean, binary_targets

RUNS = ('v14_candidate_res320_seed2026', 'v14_reference_res224_seed2026')
REGISTRY_SHA256 = 'da8fc216144e5f233c4e58363632c53d7a43d1d8c6e07300ca366d965f7aafce'
FROZEN_PLAN_SHA256 = '43351a86118e6465a0e24939faddecd9b09d8c8af49d1122c8f0f47f197749e9'
ROLES = ('train', 'validation', 'calibration')
FIXTURE = 'synthetic_fixture_not_real_v15_results'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified_path(root, name, digest):
    """Reject traversal/symlinks and verify bytes before parsing any artifact."""
    root = Path(root).resolve()
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Artifact path must stay inside its input root')
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('Artifact symlinks forbidden')
    if not path.is_file() or not isinstance(digest, str) or len(digest) != 64 or sha256(path) != digest:
        raise ValueError('Artifact SHA256 mismatch or missing file: ' + name)
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def verify_artifacts(root, expected):
    """All listed bytes, including opaque best.pt, must match trusted digests."""
    return {name: verified_path(root, name, digest) for name, digest in expected.items()}


def validate_original_role(role, original, rows):
    """Match original v4 eye rows without reducing identities to patient IDs."""
    if original.get('role') != role or original.get('schema') != 'phase2_eye_rows_v1':
        raise ValueError('Original frozen manifest role/schema mismatch')
    def identity(row):
        return (row['component_id'], row['path'], None if role == 'train' else row['labels'])
    expected = {(r['patient_id'], r['eye']): identity(r) for r in original['rows']}
    actual = {(r['patient_id'], r['eye']): identity(r) for r in rows}
    if len(expected) != len(original['rows']) or len(actual) != len(rows) or expected != actual:
        raise ValueError('Derived rows differ from frozen original manifest')


def audit_roles(roles, *, fixture=False):
    if set(roles) != set(ROLES):
        raise ValueError('Only train, validation and calibration roles permitted')
    patients, components, image_ids = set(), set(), set()
    counts = {'train': 2275, 'validation': 525, 'calibration': 350}
    for role, rows in roles.items():
        pairs = {}
        for row in rows:
            for key in ('patient_id', 'component_id', 'path'):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError('Nonempty patient/component/image identities required')
            if fixture and not row['patient_id'].startswith('fixture-'):
                raise ValueError('Synthetic fixture IDs required')
            if row['path'] in image_ids or row.get('eye') not in ('left', 'right'):
                raise ValueError('Duplicate image identity or invalid eye')
            image_ids.add(row['path'])
            if 'role' in row and row['role'] != role:
                raise ValueError('Row role mismatch')
            if role == 'train':
                if 'labels' in row:
                    raise ValueError('TRAIN quality metadata must be label-free')
            else:
                binary_targets([row.get('labels')])
            pairs.setdefault(row['patient_id'], []).append(row)
        groups = set()
        if not pairs or (not fixture and len(pairs) != counts[role]):
            raise ValueError('Frozen role patient count mismatch')
        for pid, pair in pairs.items():
            if len(pair) != 2 or {r['eye'] for r in pair} != {'left', 'right'}:
                raise ValueError('Both eyes required exactly once')
            if pair[0]['component_id'] != pair[1]['component_id']:
                raise ValueError('Patient spans components')
            if role != 'train' and pair[0]['labels'] != pair[1]['labels']:
                raise ValueError('Inherited patient labels disagree between eyes')
            groups.add(pair[0]['component_id'])
        if patients.intersection(pairs) or components.intersection(groups):
            raise ValueError('Patient/component crosses roles')
        patients.update(pairs)
        components.update(groups)


def align_role(rows, images, patients, passes, *, role):
    for field, key in (('path', 'image_ids'), ('patient_id', 'patient_ids'), ('eye', 'eyes')):
        if images[key].tolist() != [row[field] for row in rows]:
            raise ValueError('Image row alignment mismatch')
    aggregated = aggregate_patient_mean(
        [r['patient_id'] for r in rows], [r['eye'] for r in rows],
        [r['labels'] for r in rows], images['probabilities'],
        [r['component_id'] for r in rows])
    for key in ('patient_ids', 'component_ids', 'probabilities'):
        if not np.array_equal(patients[key], aggregated[key]):
            raise ValueError('Patient aggregation/alignment mismatch')
    for source in (images, patients):
        if 'labels' in source:
            wanted = [r['labels'] for r in rows] if source is images else aggregated['labels']
            if not np.array_equal(source['labels'], wanted):
                raise ValueError('Prediction labels disagree with authoritative role metadata')
    n = len(aggregated['labels'])
    samples = passes['probabilities']
    start = 62026 if role == 'validation' else 72026
    if (samples.shape != (20, n, 8) or not np.isfinite(samples).all()
            or np.any((samples < 0) | (samples > 1))
            or passes['patient_ids'].tolist() != aggregated['patient_ids']
            or passes['seeds'].tolist() != list(range(start, start + 20))):
        raise ValueError('MC pass/seed/patient alignment mismatch')
    return {**aggregated, 'passes': samples}


def authorize(manifest_digest, protocol_digest, approval, *, fixture):
    """Fail closed before reading a manifest or private artifact paths."""
    required = {'approved': True, 'scope': 'v15_cpu_shared_temperature_analysis',
                'manifest_sha256': manifest_digest, 'protocol_sha256': protocol_digest,
                'data_kind': FIXTURE if fixture else 'real_v14', 'run_ids': list(RUNS)}
    if not approval or any(approval.get(k) != v for k, v in required.items()):
        raise PermissionError('Separate exact V15 analysis authorization required before input access')
    if not fixture and not approval.get('user_approval_reference'):
        raise PermissionError('Real-data fitting requires separate explicit user authorization')


def load_bundle(manifest_path, manifest_digest, protocol_digest, approval, *, fixture=False):
    """Load only explicitly enumerated files. No recursive discovery or downloads.

    A future real-data approval must bind the prepared, reviewed analysis manifest.
    Fixture registries are accepted only by the explicitly marked fixture path.
    """
    authorize(manifest_digest, protocol_digest, approval, fixture=fixture)
    manifest_path = Path(manifest_path)
    manifest = read_json(verified_path(manifest_path.parent, manifest_path.name, manifest_digest))
    if (manifest.get('class_order') != CLASSES or manifest.get('run_ids') != list(RUNS)
            or manifest.get('protocol_sha256') != protocol_digest
            or manifest.get('data_kind') != (FIXTURE if fixture else 'real_v14')
            or set(manifest.get('roles', {})) != set(ROLES)):
        raise ValueError('Manifest class order, protocol, matrix or roles mismatch')
    root = manifest_path.parent
    if fixture:
        registry = manifest['fixture_registry']
        parents = manifest['parent_manifest_sha256']
    else:
        registry_path = verified_path(root, manifest['registry_file'], REGISTRY_SHA256)
        frozen_path = verified_path(root, manifest['frozen_plan_file'], FROZEN_PLAN_SHA256)
        registry = read_json(registry_path)
        parents = read_json(frozen_path)['registry']['role_manifest_sha256']
    if [f['run_id'] for f in registry['fits']] != list(RUNS):
        raise ValueError('Exact V14 matrix required')
    roles = {}
    for role, item in manifest['roles'].items():
        if item['parent_manifest_sha256'] != parents[role]:
            raise ValueError('Frozen parent role manifest mismatch')
        roles[role] = read_json(verified_path(root, item['file'], item['sha256']))
    audit_roles(roles, fixture=fixture)
    if not fixture:
        # Original private v4 role manifests must be supplied and hashed too;
        # an asserted parent digest alone cannot authenticate derived label rows.
        for role, item in manifest['roles'].items():
            original = read_json(verified_path(root, item['original_file'], parents[role]))
            validate_original_role(role, original, roles[role])
    if set(manifest['runs']) != set(RUNS):
        raise ValueError('Both candidates required exactly once')
    loaded = {}
    for fit in registry['fits']:
        run_id = fit['run_id']; item = manifest['runs'][run_id]
        if not all(fit.get(k) is True for k in ('strict_best_reload', 'model_state_invariance', 'post_freeze_calibration_inference')):
            raise ValueError('Checkpoint freeze/state evidence required')
        if item['kernel_version'] != fit['kernel_version']:
            raise ValueError('Pinned V14 output version mismatch')
        runroot = root / item['directory']
        # Validate directory containment and each parent against symlink escape.
        verified_path(root, str(Path(item['directory']) / 'best.pt'), fit['artifact_sha256']['best.pt'])
        paths = verify_artifacts(runroot, fit['artifact_sha256'])
        stage = read_json(verified_path(runroot, 'stage-report.json', fit['report_sha256']))
        selected = read_json(paths['selected-checkpoint.json'])
        spec = read_json(paths['experiment-spec.json'])
        if (stage.get('status') != 'complete' or stage.get('run_id') != run_id
                or stage.get('strict_best_reload') is not True
                or stage.get('model_state_invariance') is not True
                or stage.get('calibration_inference_after_freeze') is not True
                or stage.get('final_evaluation_access') is not False
                or stage.get('calibration_labels_accessed') is not False
                or stage.get('output_hashes') != fit['artifact_sha256']
                or stage.get('selected_checkpoint_sha256') != fit['artifact_sha256']['best.pt']
                or stage.get('provenance') != selected.get('provenance')):
            raise ValueError('Completed V14 inference report provenance mismatch')
        if (selected['checkpoint_sha256'] != fit['artifact_sha256']['best.pt']
                or selected['selection_role'] != 'validation'
                or selected['decision_threshold'] != .5
                or selected['patient_aggregation'] != 'mean_left_right_sigmoid_before_threshold'
                or spec['run_id'] != run_id or spec['config']['class_order'] != CLASSES
                or spec['role_manifest_sha256'] != parents
                or spec['source_commit'] != fit['source_commit']
                or selected['provenance']['run_id'] != run_id
                or selected['provenance']['image_size'] != (320 if run_id == RUNS[0] else 224)
                or spec['image_size'] != selected['provenance']['image_size']):
            raise ValueError('Checkpoint/specification provenance mismatch')
        quality = load_npz(paths['quality-features.npz'])
        data = {}
        for role in ROLES:
            for field in ('path', 'patient_id', 'eye', 'component_id'):
                if quality[role + '_' + field].tolist() != [r[field] for r in roles[role]]:
                    raise ValueError('Quality row alignment mismatch')
            features = quality[role]
            if features.shape != (len(roles[role]), 3) or not np.isfinite(features).all() or (features < 0).any():
                raise ValueError('Finite nonnegative aligned technical quality required')
            if role != 'train':
                # V14 emits predictions sorted by patient then eye, independently
                # of quality feature order. Validate each against its own order.
                ordered = sorted(roles[role], key=lambda r: (r['patient_id'], r['eye']))
                data[role] = align_role(ordered, load_npz(paths[role+'-images.npz']),
                                       load_npz(paths[role+'-patients.npz']),
                                       load_npz(paths[role+'-passes.npz']), role=role)
        loaded[run_id] = {'data': data, 'quality': quality, 'roles': roles, 'fit': fit}
    for role in ('validation', 'calibration'):
        a, b = [loaded[r]['data'][role] for r in RUNS]
        for key in ('patient_ids', 'component_ids', 'labels'):
            if not np.array_equal(a[key], b[key]):
                raise ValueError('Candidate patient/component/target panels differ')
    return loaded
