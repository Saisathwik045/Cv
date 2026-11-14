"""Offline V4 byte reconciliation. No network, image, model or fitting operations."""
import hashlib
import json
import os
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[2]
ROLES = ('train', 'validation', 'calibration')
CLASSES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
PROTOCOL = 'development-content-containment-v4'
FIELDS = {'patient_id', 'eye', 'labels', 'component_id', 'path', 'raw_sha256'}


class Blocked(ValueError):
    """Only fixed, record-free diagnostic codes leave the workflow."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def plan():
    """Read public Git records only; never discover private artifacts."""
    records = {
        'frozen': 'results/candidate-v14/frozen-plan.json',
        'v13': 'results/resolution-v13/preparation-receipt.json',
        'queue': 'results/resolution-v13/queue.json',
        'v6': 'results/reliability-v6/run-registry.json',
    }
    raw = {k: (ROOT / v).read_bytes() for k, v in records.items()}
    if digest(raw['frozen']) != '43351a86118e6465a0e24939faddecd9b09d8c8af49d1122c8f0f47f197749e9':
        raise Blocked('frozen_plan_identity')
    data = {k: json.loads(v) for k, v in raw.items()}
    # Other public records are hash-bound into the reviewable plan/approval.
    v6 = data['v6']['registry']['stages']['inputs']
    sources = {}
    for role in ROLES:
        sources[role] = {
            'kernel_version': data['v13']['kernel_version'] if role == 'train' else v6['kernel_version'],
            'file': f'prepared/{role}-rows.json',
            'sha256': data['queue']['registry']['cache']['rows_sha256'] if role == 'train'
                      else v6['artifact_hashes'][f'{role}-rows.json'],
        }
    return {
        'schema': 'v15_original_manifest_reconciliation_v1',
        'class_order': CLASSES,
        'protocol': PROTOCOL,
        'sources': sources,
        'expected': data['frozen']['registry']['role_manifest_sha256'],
        'patient_counts': {'train': 2275, 'validation': 525, 'calibration': 350},
        'public_records': {records[k]: digest(v) for k, v in raw.items()},
        'source_code_sha256': {name: digest((ROOT / name).read_bytes()) for name in (
            'src/trustretina/data/contained_release.py',
            'src/trustretina/runner/full_inputs.py',
            'src/trustretina/runner/reliability_inputs.py',
            'scripts/resolution_prepare_runtime.py',
            'scripts/trustretina_v15/reconciliation.py',
            'scripts/reconcile_v15_manifests.py')},
        'publisher': 'src/trustretina/data/contained_release.py:publish_release',
        'serialization': 'json.dumps(payload, sort_keys=True).encode(utf-8); no newline',
    }


def serialize(role, rows):
    """Exactly the original publisher: retain array order and all six row fields."""
    return json.dumps({'schema': 'phase2_eye_rows_v1', 'protocol': PROTOCOL,
                       'role': role, 'rows': rows}, sort_keys=True).encode('utf-8')


def audit(roles, expected):
    if expected['class_order'] != CLASSES or set(roles) != set(ROLES):
        raise Blocked('class_order_or_roles')
    seen_patients, seen_components, seen_paths = set(), set(), set()
    for role in ROLES:
        rows = roles[role]
        if not isinstance(rows, list) or not rows or len(rows) % 2:
            raise Blocked('missing_eyes')
        patients, groups = set(), {}
        for i in range(0, len(rows), 2):
            pair = rows[i:i+2]
            for row in pair:
                if not isinstance(row, dict) or set(row) != FIELDS:
                    raise Blocked('row_schema_or_role')
                if any(not isinstance(row[k], str) or not row[k] for k in FIELDS - {'labels'}):
                    raise Blocked('identity_type')
                if (not isinstance(row['labels'], list) or len(row['labels']) != 8
                        or any(type(v) is not int or v not in (0, 1) for v in row['labels'])):
                    raise Blocked('labels')
                if (len(row['raw_sha256']) != 64
                        or any(c not in '0123456789abcdef' for c in row['raw_sha256'])):
                    raise Blocked('raw_digest')
                if row['path'] in seen_paths:
                    raise Blocked('duplicate_image')
                seen_paths.add(row['path'])
            left, right = pair
            if [r['eye'] for r in pair] != ['left', 'right']:
                raise Blocked('eye_order')
            pid = left['patient_id']
            if pid != right['patient_id'] or pid in patients or pid in seen_patients:
                raise Blocked('duplicate_or_split_patient')
            if left['labels'] != right['labels']:
                raise Blocked('eye_label_mismatch')
            if left['component_id'] != right['component_id']:
                raise Blocked('eye_component_mismatch')
            patients.add(pid)
            groups.setdefault(left['component_id'], []).append(pid)
        if len(patients) != expected['patient_counts'][role]:
            raise Blocked('patient_count')
        for component, members in groups.items():
            if component in seen_components or component != canonical(sorted(members)):
                raise Blocked('containment_component')
        seen_patients.update(patients)
        seen_components.update(groups)


def no_symlinks(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise Blocked('symlink')
    return path


def private_dir(path):
    path = no_symlinks(path)
    info = path.stat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700 or path.resolve().is_relative_to(ROOT)):
        raise Blocked('private_directory')
    return path


def private_write(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def unique_object(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise Blocked('duplicate_json_key')
        result[k] = v
    return result


def reconcile(input_root, output, expected):
    """Core for authorized local use or synthetic tests; no download capability.

    Every source is hashed before any row is parsed. Candidate bytes remain in
    memory until ALL output hashes pass. Only record-free diagnostics are saved.
    """
    input_root = private_dir(input_root)
    output = no_symlinks(output)
    private_dir(output.parent)
    output.mkdir(mode=0o700, exist_ok=False)
    output.chmod(0o700)
    report = {'status': 'blocked', 'plan_sha256': canonical(expected),
              'input_sha256': {}, 'output_sha256': {}, 'v15_fitting_authorized': False}
    try:
        blobs = {}
        for role in ROLES:
            # Fixed local layout, irrespective of untrusted remote/source paths.
            path = no_symlinks(input_root / f'{role}-rows.json')
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or stat.S_IMODE(info.st_mode) != 0o600):
                    raise Blocked('private_input_file')
                blobs[role] = stream.read()
            report['input_sha256'][role] = digest(blobs[role])
        if any(report['input_sha256'][r] != expected['sources'][r]['sha256'] for r in ROLES):
            raise Blocked('input_hash_mismatch')
        roles = {r: json.loads(blobs[r], object_pairs_hook=unique_object) for r in ROLES}
        audit(roles, expected)
        candidates = {r: serialize(r, roles[r]) for r in ROLES}
        report['output_sha256'] = {r: digest(v) for r, v in candidates.items()}
        if report['output_sha256'] != expected['expected']:
            raise Blocked('original_manifest_hash_mismatch')
        # A single directory rename publishes the three-file bundle together.
        stage = output / '.verified'
        stage.mkdir(mode=0o700)
        stage.chmod(0o700)
        for role, blob in candidates.items():
            private_write(stage / f'{role}.json', blob)
        if any(digest((stage / f'{role}.json').read_bytes()) != expected['expected'][role]
               for role in ROLES):
            raise Blocked('staged_file_hash_mismatch')
        report['status'] = 'verified'
    except Exception as error:
        # Never save exception messages, input paths, rows or candidate bytes.
        report['failure_code'] = str(error) if isinstance(error, Blocked) else 'input_or_io_failure'
        raise Blocked(report['failure_code']) from None
    finally:
        private_write(output / 'reconciliation.json', json.dumps(report, sort_keys=True, indent=2).encode())
    stage.rename(output / 'manifests')
    return report


def authorized_reconcile(input_root, output, approval):
    expected = plan()
    required = {'approved': True, 'scope': 'v15_private_original_manifest_reconciliation',
                'plan_sha256': canonical(expected), 'input_root': str(Path(input_root).absolute()),
                'output': str(Path(output).absolute()), 'calibration_fitting_authorized': False}
    if (any(approval.get(k) != v for k, v in required.items())
            or not isinstance(approval.get('user_approval_reference'), str)
            or not approval['user_approval_reference'].strip()):
        raise Blocked('separate_reconstruction_approval_required')
    return reconcile(input_root, output, expected)
