"""Versioned conservative duplicate containment; no clinical adjudication or deletion."""
import math

POLICY = 'conservative_patient_components_v1'
METHOD = 'dhash64-horizontal-lanczos-9x8-v1'


def candidates(patients, threshold):
    """Enumerate every exact or threshold candidate once in stable input order."""
    entries = [(p, eye, r, int(r['dhash'], 16)) for p in patients
               for eye, r in sorted(p['eyes'].items())]
    pairs = []
    for i, (patient, eye, record, dhash) in enumerate(entries):
        for other, other_eye, old, old_hash in entries[:i]:
            distance = (dhash ^ old_hash).bit_count()
            raw = record['sha256'] == old['sha256']
            pixels = record['pixel_sha256'] == old['pixel_sha256']
            if not (raw or pixels or distance <= threshold):
                continue
            def endpoint(p, e, r):
                return {'patient_id': p['patient_id'], 'eye': e, 'path': r['path'],
                        'sha256': r['sha256'], 'pixel_sha256': r['pixel_sha256'],
                        'dhash': r['dhash'], 'width': r['width'], 'height': r['height'],
                        'patient_labels': p['labels']}
            pairs.append({'a': endpoint(other, other_eye, old),
                          'b': endpoint(patient, eye, record),
                          'file_equal': raw, 'pixel_equal': pixels,
                          'dhash_distance': distance, 'dhash_similarity': 1 - distance / 64,
                          'detection_method': METHOD, 'hamming_threshold': threshold,
                          'conflicting_patient_labels': patient['labels'] != other['labels'],
                          'classification': 'confirmed_exact' if raw or pixels else 'unresolved_perceptual_candidate',
                          'decision': 'co_group',
                          'reason': ('Identical raw bytes or EXIF-oriented decoded RGB pixels; retain original labels and images'
                                     if raw or pixels else 'Perceptual match is not proof of duplication; conservatively contain possible leakage'),
                          'clinical_adjudication': False})
    return pairs


def components(patients, pairs):
    """Connected patient components include both eyes and all transitive edges."""
    parent = {p['patient_id']: p['patient_id'] for p in patients}
    def find(pid):
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]
            pid = parent[pid]
        return pid
    for pair in pairs:
        a, b = (pair[k]['patient_id'] for k in ('a', 'b'))
        ra, rb = find(a), find(b)
        parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for pid in sorted(parent):
        groups.setdefault(find(pid), []).append(pid)
    return sorted(groups.values(), key=lambda ids: ids[0])


def comparison(root, pair):
    """Descriptive aligned-thumbnail evidence; never a decision threshold."""
    from PIL import Image, ImageOps, ImageChops, ImageStat
    from .validation import file_hash, DataValidationError
    images, metadata = [], []
    for key in ('a', 'b'):
        if file_hash(root / pair[key]['path']) != pair[key]['sha256']:
            raise DataValidationError('Image changed before duplicate comparison')
        with Image.open(root / pair[key]['path']) as source:
            rgb = ImageOps.exif_transpose(source).convert('RGB')
            rgb = rgb.resize((64, 64), Image.Resampling.LANCZOS)
            gray = rgb.convert('L')
            hist = gray.histogram()
            metadata.append({'gray_stddev_64': ImageStat.Stat(gray).stddev[0],
                             'entropy_bits_64': gray.entropy(),
                             'dark_fraction_64': sum(hist[:16]) / 4096})
            images.append(rgb)
        if file_hash(root / pair[key]['path']) != pair[key]['sha256']:
            raise DataValidationError('Image changed during duplicate comparison')
    delta = ImageChops.difference(*images)
    stats = ImageStat.Stat(delta)
    mse = sum(stats.sum2) / (64 * 64 * 3)
    x, y = [list(im.tobytes()) for im in images]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    correlation = (sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)
                   if vx and vy else None)
    return {'method': 'EXIF RGB resized independently to 64x64 LANCZOS; descriptive only',
            'rgb_mae_64': sum(stats.mean) / 3,
            'psnr_db_64': 10 * math.log10(255 ** 2 / mse) if mse else None,
            'correlation_64': correlation, 'resized_rgb_equal_64': x == y,
            'a': metadata[0], 'b': metadata[1],
            'limitations': 'No registration/crop search; entropy and contrast are descriptive, not clinical gradability or duplicate proof'}


def adjudicate(patients, threshold, root=None):
    pairs = candidates(patients, threshold)
    if root is not None:
        for pair in pairs:
            pair['image_comparison'] = comparison(root, pair)
    groups = components(patients, pairs)
    return {'schema_version': 1, 'policy': POLICY, 'detection_method': METHOD,
            'hamming_threshold': threshold, 'pairs': pairs, 'patient_groups': groups,
            'counts': {'candidate_pairs': len(pairs),
                       'confirmed_exact_pairs': sum(p['classification'] == 'confirmed_exact' for p in pairs),
                       'unresolved_pairs': sum(p['classification'] != 'confirmed_exact' for p in pairs),
                       'conflicting_annotation_pairs': sum(p['conflicting_patient_labels'] for p in pairs),
                       'patient_components': len(groups),
                       'nontrivial_components': sum(len(g) > 1 for g in groups),
                       'largest_component_patients': max(map(len, groups), default=0)},
            'annotation_policy': 'Patient-derived labels retained independently even when connected images have different annotations; report conflicts, never infer eye-specific labels',
            'clinical_adjudication': False, 'images_deleted': 0, 'labels_changed': 0}


def verified_components(inventory):
    """Recompute all image edges: a removed or edited adjudication cannot hide leakage."""
    from .validation import DataValidationError
    patients = inventory['patients']
    if 'duplicate_adjudication' not in inventory:
        raise DataValidationError('Duplicate adjudication is required before splitting')
    if inventory.get('duplicate_policy') not in ('reject', POLICY):
        raise DataValidationError('Unknown duplicate handling policy in inventory')
    saved = inventory['duplicate_adjudication']
    expected = adjudicate(patients, inventory['near_duplicate_hamming'])
    evidence_free = dict(saved)
    evidence_free['pairs'] = [{k: v for k, v in pair.items() if k != 'image_comparison'}
                              for pair in saved['pairs']]
    if inventory['duplicate_policy'] == 'reject' and expected['pairs']:
        raise DataValidationError('Unadjudicated duplicate candidates under reject policy')
    if expected != evidence_free:
        raise DataValidationError('Duplicate adjudication does not match all image candidate edges')
    return expected['patient_groups']
