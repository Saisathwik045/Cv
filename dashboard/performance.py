"""Bounded original-model CPU timing on generated non-patient inputs only."""
import math
import statistics
import time

from dashboard.guided import synthetic_pair


def summarize(values):
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Invalid elapsed measurements')
    return {'samples_seconds': values, 'count': len(values), 'minimum_seconds': min(values),
            'median_seconds': statistics.median(values), 'maximum_seconds': max(values)}


def check_prediction(result, contract):
    expected = {'device': 'cpu', 'model': 'convnext_tiny', 'best_epoch': 2,
                'class_order': list('NDGCAHMO'), 'mc_passes': 20, 'threshold': .5,
                'decision': 'REVIEW REQUIRED', 'temperature': contract['temperature'],
                'checkpoint_sha256': contract['checkpoint_sha256'],
                'preprocessing_sha256': contract['preprocessing_sha256']}
    if any(result[k] != value for k, value in expected.items()):
        raise ValueError('Original CPU inference contract required')
    for key in ('raw_probabilities', 'calibrated_probabilities'):
        if list(result[key]) != list('NDGCAHMO') or not all(
                math.isfinite(x) and 0 <= x <= 1 for x in result[key].values()):
            raise ValueError('Invalid operational output')


def measure(factory, contract, state_digest, *, clock=time.perf_counter):
    """One load, four predictions and one explanation; never retain predictions."""
    pair = synthetic_pair()
    start = clock()
    service = factory()
    load_seconds = clock() - start
    original_state = service.state_sha256
    durations = []
    for _ in range(4):
        start = clock()
        result = service.predict(*pair)
        elapsed = clock() - start
        check_prediction(result, contract)
        durations.append(elapsed)
        del result
    start = clock()
    cam, overlays = service.explain(*pair, 1)
    explanation_seconds = clock() - start
    if len(overlays) != 2:
        raise ValueError('Both explanation overlays required')
    del cam, overlays
    if state_digest(service.model) != original_state or any(m.training for m in service.model.modules()):
        raise ValueError('Frozen model state changed')
    return {'load_including_internal_bundle_verification_seconds': summarize([load_seconds]),
            'first_prediction_seconds': summarize(durations[:1]),
            'repeated_prediction_seconds': summarize(durations[1:]),
            'paired_D_gradcam_seconds': summarize([explanation_seconds]),
            'calls': {'model_loads': 1, 'predictions': 4, 'paired_explanations': 1},
            'frozen_state_unchanged': True}
