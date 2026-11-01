#!/usr/bin/env python3
"""Bounded CPU audit of existing TRAIN-only predictions; no inference or fitting."""
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trustretina.data.validation import file_hash, canonical_hash

CLASSES = ('N', 'D', 'G', 'C', 'A', 'H', 'M', 'O')
ARMS = ('ratio_bce', 'sqrt_ratio_bce')
PRIVATE = ROOT / 'artifacts/audit-v12'
PUBLIC = ROOT / 'results/development-audit-v12'


def rank_key(pid):
    return hashlib.sha256(('trustretina-audit-v12|2026|' + str(pid)).encode()).hexdigest()


def outcomes(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Require aligned binary targets and finite probabilities')
    return np.where(y == 1, np.where(p >= .5, 'TP', 'FN'), np.where(p >= .5, 'FP', 'TN'))


def select_cases(ids, groups, y, p):
    if len(set(ids)) != len(ids) or len(ids) != len(groups) or y.shape != (len(ids), 8):
        raise ValueError('Invalid patient/group alignment')
    selected = []
    for arm in ARMS:
        for label in ('H', 'O'):
            j = CLASSES.index(label)
            status = outcomes(y[:, j], p[arm][:, j])
            for outcome in ('TP', 'FP', 'FN', 'TN'):
                seen = set()
                for i in sorted(np.flatnonzero(status == outcome), key=lambda i: rank_key(ids[i])):
                    if groups[i] in seen:
                        continue
                    seen.add(groups[i])
                    selected.append({'arm': arm, 'label': label, 'outcome': outcome, 'index': int(i)})
                    if len(seen) == 2:
                        break
    return selected


def pixel_features(rgb):
    x = np.asarray(rgb)
    if x.shape != (224, 224, 3) or x.dtype != np.uint8:
        raise ValueError('Require original cached RGB uint8 224 input')
    padding = np.all(x == np.asarray([124, 116, 104]), axis=2)
    gray = x.astype(np.float64).mean(2) / 255
    valid = ~padding
    foreground = valid & (gray > 8 / 255)
    energy = []
    for axis in (0, 1):
        mask = (foreground[1:] & foreground[:-1]) if axis == 0 else (foreground[:, 1:] & foreground[:, :-1])
        differences = np.diff(gray, axis=axis)[mask]
        if len(differences):
            energy.append(float(np.mean(differences ** 2)))
    return {'mean_color_fraction': float(padding.mean()),
            'nonpadding_dark_fraction': float(((gray <= 8 / 255) & valid).sum() / max(1, valid.sum())),
            'foreground_contrast': float(gray[foreground].std()) if foreground.any() else None,
            'foreground_gradient_energy': float(sum(energy)) if energy else None}


def quantiles(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    return {'n': len(a), 'q25_median_q75': np.quantile(a, [.25, .5, .75]).tolist() if len(a) else None}


def counts(status):
    c = Counter(status.tolist())
    tp, fp, fn = [c[k] for k in ('TP', 'FP', 'FN')]
    return {'n': len(status), **{k: c[k] for k in ('TP', 'FP', 'FN', 'TN')},
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def positive_support_summary(status):
    """Recall on true positives only; precision/F1 on this subset are misleading."""
    c = counts(status)
    if c['FP'] or c['TN']:
        raise ValueError('Positive-support summary requires true positives only')
    return {k: c[k] for k in ('n', 'TP', 'FN', 'recall')}


def pinned_output(bundle, report_name, relative):
    bundle = Path(bundle)
    verification = json.loads((bundle / 'verification.json').read_text())
    submission = json.loads((bundle / 'submission.json').read_text())
    report_path = bundle / 'download' / report_name
    if not verification.get('integrity_verified') or file_hash(report_path) != verification['artifact_sha256']:
        raise ValueError('Source metadata report integrity mismatch')
    report = json.loads(report_path.read_text())
    for k in ('nonce', 'source_commit'):
        if verification[k] != submission[k] or report[k] != submission[k]:
            raise ValueError('Source metadata receipt identity mismatch')
    output = bundle / 'download' / relative
    if file_hash(output) != report['output_hashes'][relative]:
        raise ValueError('Source metadata output mismatch')
    return output


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    started = time.monotonic()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    if (PUBLIC / 'audit.json').exists():
        raise ValueError('Audit already complete; reuse existing report')
    registry = json.loads((ROOT / 'artifacts/runner/nested-v11/registry.json').read_text())
    provenance = {}
    def checked(path, digest=None):
        path = Path(path)
        actual = file_hash(path)
        if digest is not None and actual != digest:
            raise ValueError('Audit input checksum changed: ' + str(path))
        provenance[str(path.relative_to(ROOT))] = actual
        return path
    cache = ROOT / registry['cache']['bundle'] / 'download/prepared'
    inputs = json.loads(checked(cache / 'inputs.json', registry['cache']['inputs_sha256']).read_text())
    item = inputs['roles']['train']
    rows = json.loads(checked(cache / item['rows_file'], item['rows_sha256']).read_text())
    images_path = checked(cache / item['images_file'], item['images_sha256'])
    images = np.load(images_path, mmap_mode='r', allow_pickle=False)
    if images.shape != (4550, 3, 224, 224) or images.dtype != np.uint8:
        raise ValueError('Wrong TRAIN-only image cache')
    by_patient = {}
    for i, r in enumerate(rows):
        pair = by_patient.setdefault(str(r['patient_id']), {})
        if r['eye'] in pair:
            raise ValueError('Repeated eye')
        pair[r['eye']] = (i, r)
    ids = sorted(by_patient)
    if len(ids) != 2275:
        raise ValueError('Wrong training cohort')
    groups, y = [], []
    for pid in ids:
        pair = by_patient[pid]
        if set(pair) != {'left', 'right'}:
            raise ValueError('Both eyes required')
        left, right = pair['left'][1], pair['right'][1]
        if left['labels'] != right['labels'] or left['component_id'] != right['component_id']:
            raise ValueError('Repeated patient-target/group mismatch')
        y.append(left['labels']); groups.append(str(left['component_id']))
    y = np.asarray(y)
    lookup = {pid: i for i, pid in enumerate(ids)}
    predictions = {a: np.full((len(ids), 8), np.nan) for a in ARMS}
    eyes = {a: np.full((len(ids), 2, 8), np.nan) for a in ARMS}
    fold_ids = np.full(len(ids), -1)
    manifest = json.loads(checked(ROOT / 'artifacts/runner/nested-v11/fold-manifest.json').read_text())
    if manifest['sha256'] != registry['fold_manifest_sha256'] or manifest['sha256'] != canonical_hash({k:v for k,v in manifest.items() if k!='sha256'}):
        raise ValueError('Frozen fold manifest identity changed')
    for run in registry['runs']:
        if run['status'] != 'complete':
            raise ValueError('Complete matrix required')
        bundle = ROOT / run['attempts'][-1]['bundle']
        v = json.loads((bundle / 'verification.json').read_text())
        s = json.loads((bundle / 'submission.json').read_text())
        if v.get('integrity_verified') is not True or any(v[k] != s[k] for k in ('nonce', 'source_sha256', 'kernel_version')):
            raise ValueError('Wrong saved receipts')
        report_path = checked(bundle / 'download' / v['report_relative_path'], v['artifact_sha256'])
        report = json.loads(report_path.read_text())
        if report['nonce'] != s['nonce'] or report['run_id'] != run['id'] or report['plan_sha256'] != registry['plan_sha256']:
            raise ValueError('Wrong saved report')
        folder = report_path.parent
        pp = checked(folder / 'outer-patients.npz', report['output_hashes']['outer-patients.npz'])
        ep = checked(folder / 'outer-images.npz', report['output_hashes']['outer-images.npz'])
        expected = {str(rows[i]['patient_id']) for i in manifest['folds'][run['outer_fold']]['outer_indices']}
        with np.load(pp, allow_pickle=False) as d:
            if set(d['patient_ids']) != expected:
                raise ValueError('Wrong outer coverage')
            for pid, label, prob, group in zip(d['patient_ids'], d['labels'], d['probabilities'], d['component_ids']):
                i = lookup[str(pid)]
                if not np.array_equal(y[i], label) or groups[i] != str(group) or np.isfinite(predictions[run['arm']][i]).any():
                    raise ValueError('Mismatched/repeated outer rows')
                predictions[run['arm']][i] = prob
                fold_ids[i] = run['outer_fold']
        with np.load(ep, allow_pickle=False) as d:
            for pid, eye, label, prob in zip(d['patient_ids'], d['eyes'], d['labels'], d['probabilities']):
                i = lookup[str(pid)]; j = ('left', 'right').index(str(eye))
                if not np.array_equal(y[i], label) or np.isfinite(eyes[run['arm']][i, j]).any():
                    raise ValueError('Mismatched/repeated eye prediction')
                eyes[run['arm']][i, j] = prob
        if time.monotonic() - started > 1800:
            raise TimeoutError('CPU audit budget exhausted')
    for arm in ARMS:
        if not np.isfinite(eyes[arm]).all() or not np.isfinite(predictions[arm]).all():
            raise ValueError('Missing predictions')
        np.testing.assert_allclose(predictions[arm], eyes[arm].mean(1), rtol=0, atol=1e-7)
    selected = select_cases(ids, groups, y, predictions)
    chosen = sorted({r['index'] for r in selected}, key=lambda i: rank_key(ids[i]))
    codes = {i: f'C{k+1:02}' for k, i in enumerate(chosen)}
    for r in selected:
        r.update(patient_id=ids[r['index']], component_id=groups[r['index']], case=codes[r['index']])
    selection = {'protocol_sha256': file_hash(ROOT / 'docs/DEVELOPMENT_AUDIT_V12.md'), 'strata': selected}
    if (PRIVATE / 'selection.json').exists() and json.loads((PRIVATE / 'selection.json').read_text()) != selection:
        raise ValueError('Fixed visual selection changed; preserve and investigate')
    if not (PRIVATE / 'selection.json').exists():
        write(PRIVATE / 'selection.json', selection)
    # No final/calibration rows are used: immediately filter shared historical metadata.
    metadata_path = checked(pinned_output(ROOT / 'artifacts/runner/duplicate-pair-review', 'review-report.json', 'comparisons/workbook-metadata.json'))
    metadata_payload = json.loads(metadata_path.read_text())
    keywords = {pid: metadata_payload['metadata'][pid] for pid in ids if pid in metadata_payload['metadata']}
    inventory_path = checked(pinned_output(ROOT / 'artifacts/runner/support-floor-publication', 'validation-report.json', 'validation-output/phase1/inventory.json'))
    inventory_payload = json.loads(inventory_path.read_text())
    source = {str(r['patient_id']): r for r in inventory_payload['patients'] if str(r['patient_id']) in lookup}
    if metadata_payload['workbook_sha256'] != inventory_payload['workbook']['sha256']:
        raise ValueError('Source metadata workbook mismatch')
    del metadata_payload, inventory_payload
    source_mismatches = 0
    for pid in ids:
        if pid not in source:
            raise ValueError('Missing source row')
        pair = by_patient[pid]
        source_mismatches += int(source[pid]['labels'] != y[lookup[pid]].tolist())
        for eye in ('left', 'right'):
            raw = source[pid]['eyes'][eye]; row = pair[eye][1]
            if raw['sha256'] != row['raw_sha256'] or raw['path'] != row['path']:
                raise ValueError('Source image reference/hash mismatch')
    quality = []
    geometry = []
    for pid in ids:
        q = []
        for eye in ('left', 'right'):
            index, row = by_patient[pid][eye]
            q.append(pixel_features(images[index].transpose(1, 2, 0)))
            src = source[pid]['eyes'][eye]
            w, h = src['width'], src['height']; scale = min(224/w, 224/h)
            rw, rh = max(1, round(w*scale)), max(1, round(h*scale))
            left, top = (224-rw)//2, (224-rh)//2
            x = images[index].transpose(1, 2, 0)
            mask = np.ones((224,224), dtype=bool); mask[top:top+rh,left:left+rw] = False
            if mask.any() and not np.all(x[mask] == [124,116,104]):
                raise ValueError('Padding differs from recorded geometry')
            geometry.append({'resize_short_side': min(rw,rh), 'padding_fraction': float(mask.mean()), 'long_side_downscale': max(w,h)/224})
        quality.append({k: float(np.mean([r[k] for r in q])) if all(r[k] is not None for r in q) else None for k in q[0]})
        if time.monotonic() - started > 1800:
            raise TimeoutError('CPU audit budget exhausted')
    np.savez_compressed(PRIVATE / 'patient-evidence.npz', patient_ids=np.asarray(ids), labels=y, fold=fold_ids,
                        **{a: predictions[a] for a in ARMS})
    write(PRIVATE / 'quality.json', quality)
    evidence = []
    for i in chosen:
        pid = ids[i]
        evidence.append({'case':codes[i], 'patient_id':pid, 'memberships': [{k:r[k] for k in ('arm','label','outcome')} for r in selected if r['index']==i],
                         'source_keywords':keywords.get(pid), 'labels':y[i].tolist(), 'quality':quality[i],
                         'probabilities':{a:predictions[a][i].tolist() for a in ARMS}})
    write(PRIVATE / 'selected-evidence.json', evidence)
    for page, offset in enumerate(range(0,len(chosen),4),1):
        canvas = Image.new('RGB',(1010,4*275),(245,245,245));draw=ImageDraw.Draw(canvas)
        for line,i in enumerate(chosen[offset:offset+4]):
            pid=ids[i];py=line*275
            memberships='; '.join(r['label']+' '+r['outcome']+' '+('R' if r['arm']=='ratio_bce' else 'S') for r in selected if r['index']==i)
            draw.text((8,py+2),codes[i]+' | '+memberships,fill='black')
            for e,eye in enumerate(('left','right')):
                x=images[by_patient[pid][eye][0]].transpose(1,2,0)
                canvas.paste(Image.fromarray(x),(8+e*240,py+30));draw.text((8+e*240,py+17),eye,fill='black')
            draw.text((500,py+30),'224px model inputs; technical review only',fill='black')
        canvas.save(PRIVATE/f'visual-page-{page:02}.png')
    result={'cohort':'v4 TRAIN-only / v11 outer OOF', 'patients':len(ids), 'components':len(set(groups)),
            'protocol_sha256':file_hash(ROOT/'docs/DEVELOPMENT_AUDIT_V12.md'),
            'source_consistency':{'source_rows':len(source),'label_mismatches':source_mismatches,
                                  'N_with_disease':int(np.sum((y[:,0]==1)&(y[:,1:].sum(1)>0))),
                                  'identical_patient_targets_per_eye':True, 'keyword_coverage':len(keywords),
                                  'keywords_missing':len(ids)-len(keywords), 'keyword_cache_selection_biased':True,
                                  'source_metadata_receipts_verified':True},
            'visual_selection':{'stratum_records':len(selected),'unique_patients':len(chosen),'selection_sha256':file_hash(PRIVATE/'selection.json'), 'pages':(len(chosen)+3)//4},
            'preprocessing':{k:quantiles([r[k] for r in geometry]) for k in geometry[0]}, 'classes':{},
            'provenance':provenance, 'new_model_inference':False, 'gpu_submissions':0, 'thresholds_fitted':False,
            'native_raw_images_available':False, 'historical_final_predictions_opened':False, 'calibration_predictions_opened':False}
    for label in ('H','O'):
        j=CLASSES.index(label);positive=y[:,j]==1;available=np.asarray([pid in keywords for pid in ids]);nlabels=y.sum(1)
        words=[keywords[pid] for pid in ids if pid in keywords and positive[lookup[pid]]]
        def texts(w):return tuple(str(w[k]).strip().casefold() for k in ('Left-Diagnostic Keywords','Right-Diagnostic Keywords'))
        both=[texts(w) for w in words]
        lex={}
        for term in ('normal fundus','hypertensive retinopathy'):
            matches=[sum(bool(re.search(r'\b'+re.escape(term)+r'\b',t)) for t in ts) for ts in both]
            lex[term]={'neither':matches.count(0),'one_eye':matches.count(1),'both_eyes':matches.count(2)}
        entry={'positive':int(positive.sum()),'negative':int((~positive).sum()),
               'positive_keyword_coverage':int((positive&available).sum()),
               'co_positive':{c:int(np.sum(positive&(y[:,k]==1))) for k,c in enumerate(CLASSES)},
               'single_label_positive':int(np.sum(positive&(nlabels==1))),
               'keywords':lex, 'distinct_positive_eye_keyword_strings':len({t for ts in both for t in ts}), 'arms':{}}
        for arm in ARMS:
            status=outcomes(y[:,j],predictions[arm][:,j]);e=eyes[arm][:,:,j];diff=np.abs(e[:,0]-e[:,1])
            a={'counts':counts(status), 'one_eye_positive_mean_negative_among_true_positives':int(np.sum(positive&(e.max(1)>=.5)&(predictions[arm][:,j]<.5))),
               'status':{}, 'single_label_positives':positive_support_summary(status[positive&(nlabels==1)]), 'multilabel_positives':positive_support_summary(status[positive&(nlabels>1)])}
            for st in ('TP','FP','FN','TN'):
                idx=np.flatnonzero(status==st)
                a['status'][st]={'n':len(idx),'patient_score':quantiles(predictions[arm][idx,j]),'eye_score_gap':quantiles(diff[idx]),
                                  'quality':{k:quantiles([quality[i][k] for i in idx]) for k in quality[0]}}
            entry['arms'][arm]=a
        result['classes'][label]=entry
    result['elapsed_cpu_wall_seconds']=time.monotonic()-started
    write(PUBLIC/'audit.json',result)
    write(PRIVATE/'PRIVATE_INDEX.json',{'selection':'selection.json','evidence':'selected-evidence.json','quality':'quality.json','montages':'visual-page-*.png','public_report':'results/development-audit-v12/audit.json',
                                      'no_training_or_inference':True,'protocol_commit':'c244d33','artifact_hashes':{p.name:file_hash(p) for p in PRIVATE.iterdir() if p.is_file()}})
    print(json.dumps({k:result[k] for k in ('patients','components','source_consistency','visual_selection','elapsed_cpu_wall_seconds')},indent=2))


if __name__ == '__main__':
    main()
