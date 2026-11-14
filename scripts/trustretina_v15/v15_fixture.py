"""Reproducible saved-array fixtures, never real patient or model evidence."""
from pathlib import Path

import numpy as np

from trustretina.config import CLASSES
from .v15_inputs import FIXTURE, ROLES, RUNS, sha256
from .v15 import write_json


def create_fixture(root, protocol_sha256):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(2026)
    roles = {}; items = {}; parents = {}
    for role in ROLES:
        rows = []
        for i in range(60):
            for eye in ('left', 'right'):
                row = {'patient_id': f'fixture-{role}-{i:03}', 'component_id': f'fixture-{role}-g{i:03}',
                       'path': f'{role}/{i:03}-{eye}.png', 'eye': eye}
                if role != 'train':
                    row['labels'] = [int((i+j) % 2) for j in range(8)]
                rows.append(row)
        roles[role] = rows
        write_json(root/(role+'-rows.json'), rows)
        parents[role] = sha256(root/(role+'-rows.json'))
        items[role] = {'file': role+'-rows.json', 'sha256': parents[role], 'parent_manifest_sha256': parents[role]}
    fits = []; runs = {}
    for arm, run_id in enumerate(RUNS):
        runroot = root/run_id; runroot.mkdir()
        (runroot/'best.pt').write_bytes(b'FIXTURE opaque bytes; never torch.load '+str(arm).encode())
        checkpoint = sha256(runroot/'best.pt')
        provenance = {'run_id': run_id, 'image_size': 320 if arm == 0 else 224}
        write_json(runroot/'selected-checkpoint.json', {'checkpoint_sha256': checkpoint,
                   'selection_role': 'validation', 'decision_threshold': .5,
                   'provenance': provenance,
                   'patient_aggregation': 'mean_left_right_sigmoid_before_threshold'})
        write_json(runroot/'experiment-spec.json', {'run_id': run_id, 'image_size': provenance['image_size'],
                   'config': {'class_order': CLASSES}, 'role_manifest_sha256': parents})
        quality = {}
        for role in ROLES:
            rows = roles[role]
            quality[role] = rng.random((len(rows), 3))
            for field in ('path', 'patient_id', 'eye', 'component_id'):
                quality[role+'_'+field] = np.asarray([r[field] for r in rows])
            if role == 'train':
                continue
            y = np.asarray([r['labels'] for r in rows[::2]])
            # Interior scalar optimum: retain errors, no forced improvement.
            p = np.where(y, .85, .15); p[::4] = 1-p[::4]
            if arm == 0:
                p[1::9, 5] = 1-p[1::9, 5]
            ids = [r['patient_id'] for r in rows[::2]]
            np.savez_compressed(runroot/(role+'-images.npz'), probabilities=np.repeat(p, 2, axis=0),
                                image_ids=[r['path'] for r in rows], patient_ids=[r['patient_id'] for r in rows],
                                eyes=[r['eye'] for r in rows])
            np.savez_compressed(runroot/(role+'-patients.npz'), probabilities=p,
                                patient_ids=ids, component_ids=[r['component_id'] for r in rows[::2]])
            start = 62026 if role == 'validation' else 72026
            np.savez_compressed(runroot/(role+'-passes.npz'), probabilities=np.clip(p[None]+rng.normal(0, .02, (20, 60, 8)), 0, 1),
                                patient_ids=ids, seeds=np.arange(start, start+20))
        np.savez_compressed(runroot/'quality-features.npz', **quality)
        hashes = {p.name: sha256(p) for p in runroot.iterdir()}
        write_json(runroot/'stage-report.json', {
            'status': 'complete', 'run_id': run_id, 'strict_best_reload': True,
            'source_commit': 'fixture-source',
            'model_state_invariance': True, 'calibration_inference_after_freeze': True,
            'final_evaluation_access': False, 'calibration_labels_accessed': False,
            'output_hashes': hashes, 'selected_checkpoint_sha256': checkpoint,
            'provenance': provenance, 'data_kind': FIXTURE})
        fits.append({'run_id': run_id, 'kernel_version': 'fixture/'+run_id+'/1', 'artifact_sha256': hashes,
                     'report_sha256': sha256(runroot/'stage-report.json'),
                     'source_commit': 'fixture-source', 'strict_best_reload': True,
                     'model_state_invariance': True, 'post_freeze_calibration_inference': True})
        runs[run_id] = {'directory': run_id, 'kernel_version': fits[-1]['kernel_version']}
    manifest = {'data_kind': FIXTURE, 'protocol_sha256': protocol_sha256, 'class_order': CLASSES,
                'run_ids': list(RUNS), 'roles': items, 'runs': runs, 'fixture_registry': {'fits': fits},
                'parent_manifest_sha256': parents}
    path = root/'manifest.json'; write_json(path, manifest)
    digest = sha256(path)
    approval = {'approved': True, 'scope': 'v15_cpu_shared_temperature_analysis', 'manifest_sha256': digest,
                'protocol_sha256': protocol_sha256, 'data_kind': FIXTURE, 'run_ids': list(RUNS)}
    return path, digest, approval
