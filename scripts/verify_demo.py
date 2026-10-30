#!/usr/bin/env python3
"""Bounded actual CPU verification against three prespecified development pairs. No final data."""
import io
import json
import resource
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from PIL import Image
from trustretina.inference.service import InferenceService
from trustretina.inference.bundle import CONTRACT_SHA256, digest
from trustretina.reliability.development import temperature_probabilities, canonical_quality
from trustretina.runner.reliability_runtime import state_hash
from trustretina.runner.final_analysis import frozen_quality
from trustretina.reliability.stochastic import head_patient_samples

# Set before computation; never relax these tolerances after observing results.
ATOL, RTOL = 1e-5, 1e-4
MC_INTERNAL_ATOL = 1e-7


def png(chw):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(chw).transpose(1, 2, 0)).save(buffer, format='PNG')
    return buffer.getvalue()


def main():
    torch.set_num_threads(2)
    inputs = ROOT / 'artifacts/runner/phase6/inputs/download/prepared'
    gpu = ROOT / 'artifacts/runner/phase6/inference/download/full'
    input_report = json.loads((inputs / 'stage-report.json').read_text())
    gpu_report = json.loads((gpu / 'stage-report.json').read_text())
    # Verify only consumed development artifacts, not unrelated calibration/final arrays.
    for root, report, files in ((inputs, input_report, ['validation-images.npy', 'validation-rows.json']), (gpu, gpu_report, ['validation.npz'])):
        if digest(root / 'stage-report.json') != (root / 'stage-report.sha256').read_text().strip():
            raise ValueError('Development report hash changed')
        for name in files:
            if digest(root / name) != report['output_hashes'][name]:
                raise ValueError('Saved development artifact changed')
    cache = np.load(inputs / 'validation-images.npy', mmap_mode='r', allow_pickle=False)
    rows = json.loads((inputs / 'validation-rows.json').read_text())
    with np.load(gpu / 'validation.npz', allow_pickle=False) as f:
        expected = {k: f[k] for k in ('patient_ids', 'probabilities', 'image_probabilities', 'image_ids')}
    if expected['image_ids'].tolist() != [r['path'] for r in rows]:
        raise ValueError('Saved development alignment changed')
    service = InferenceService(device='cpu')
    before = state_hash(service.model)
    selected = expected['patient_ids'][:3].tolist()  # Fixed order, independent of results.
    details = []
    last_pair = None
    for i, pid in enumerate(selected):
        idx = [next(j for j, r in enumerate(rows) if r['patient_id'] == pid and r['eye'] == eye) for eye in ('left', 'right')]
        left, right = [png(cache[j]) for j in idx]
        result = service.predict(left, right)
        raw = np.asarray(list(result['raw_probabilities'].values()))
        cal = np.asarray(list(result['calibrated_probabilities'].values()))
        np.testing.assert_allclose(raw, expected['probabilities'][i], atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(cal, temperature_probabilities(expected['probabilities'][i:i+1], service.temperature)[0], atol=ATOL, rtol=RTOL)
        tensors, _ = service.prepare(left, right)
        with torch.inference_mode():
            features = service.model.backbone(tensors)
        placeholder_rows = [{'patient_id': 'upload', 'eye': e, 'labels': [0]*8, 'component_id': 'upload'} for e in ('left', 'right')]
        direct_mc = head_patient_samples(service.model, features, placeholder_rows, passes=20, base_seed=62026)['probabilities']
        np.testing.assert_allclose(result['raw_mc_variance'], direct_mc.var(0)[0], atol=MC_INTERNAL_ATOL, rtol=0)
        q = frozen_quality(service.normalizer, np.stack([canonical_quality(cache[j]) for j in idx])).min()
        np.testing.assert_allclose(result['signals']['raw']['quality'], q, atol=1e-12, rtol=0)
        repeated = service.predict(left, right)
        np.testing.assert_array_equal(repeated['raw_mc_variance'], result['raw_mc_variance'])
        details.append({'case_index': i, 'raw_max_abs_error': float(abs(raw - expected['probabilities'][i]).max()), 'seconds': result['inference_seconds'], 'repeat_mc_exact': True})
        last_pair = (left, right)
    started = time.perf_counter()
    cam, _ = service.explain(*last_pair, 1)
    assert cam['target_index'] == 1 and cam['heatmaps'].shape == (2, 224, 224) and np.isfinite(cam['heatmaps']).all()
    assert before == state_hash(service.model) and not any(m.training for m in service.model.modules())
    report = {'contract_sha256': CONTRACT_SHA256, 'device': 'cpu', 'cuda_available': torch.cuda.is_available(), 'development_pairs': 3, 'selection': 'first three saved validation patients; fixed before inference', 'atol': ATOL, 'rtol': RTOL, 'cases': details, 'model_state_unchanged': True, 'strict_checkpoint_load': True, 'gradcam_target': 'D', 'gradcam_seconds': time.perf_counter() - started, 'peak_process_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'load_seconds': service.load_seconds, 'dependencies': result['dependencies'], 'saved_cohort_mc_parity': 'Not bitwise comparable: CPU vs CUDA RNG and single-pair vs cohort-sized masks. Exact internal frozen-helper parity and repeated-pair reproducibility verified.', 'final_evaluation_rerun': False}
    out = ROOT / 'results/demo-v8/verification.json'
    out.write_text(json.dumps(report, indent=2))
    private = ROOT / 'artifacts/demo-v8'
    private.mkdir(exist_ok=True)
    # Development-only temporary upload fixtures stay outside Git.
    (private / 'left.png').write_bytes(last_pair[0]); (private / 'right.png').write_bytes(last_pair[1])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
