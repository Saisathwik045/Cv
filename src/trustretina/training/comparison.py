"""Prespecified full-experiment decisions and metrics; no threshold fitting."""
import numpy as np
from trustretina.evaluation.metrics import multilabel_metrics, binary_targets, probabilities


def training_weights(rows):
    patients = {}
    for row in rows:
        pid = row['patient_id']
        y = row['labels']
        if pid in patients and patients[pid] != y:
            raise ValueError('Training patient labels disagree')
        patients[pid] = y
    y = binary_targets(list(patients.values()))
    positive = y.sum(0)
    negative = len(y) - positive
    if (positive == 0).any() or (negative == 0).any():
        raise ValueError('Training class missing an outcome')
    return negative / positive


def patient_loss(y, p, weights):
    y = binary_targets(y)
    p = probabilities(p, y.shape).clip(1e-7, 1-1e-7)
    return float(np.mean(-(np.asarray(weights)*y*np.log(p)+(1-y)*np.log1p(-p))))


def advance_selection(state, loss, epoch, *, min_delta, patience):
    if not np.isfinite(loss):
        raise ValueError('Non-finite selection loss')
    state = dict(state)
    best = state.get('best_loss')
    improved = best is None or loss < best
    if improved:
        state.update(best_loss=loss, best_epoch=epoch)
    reference = state.get('early_stop_reference')
    if reference is None or loss < reference-min_delta:
        state.update(early_stop_reference=loss, bad_epochs=0)
    else:
        state['bad_epochs'] = state.get('bad_epochs', 0)+1
    state['early_stopped'] = state['bad_epochs'] >= patience
    return state, improved


def comparison_metrics(y, p):
    report = multilabel_metrics(y, p, .5)
    pred = np.asarray(p) >= .5
    report['subset_exact_match_accuracy'] = float(np.all(pred == y, axis=1).mean())
    report['labelwise_binary_accuracy'] = float((pred == y).mean())
    report['hamming_loss'] = 1-report['labelwise_binary_accuracy']
    ranks = [x['auroc'] for x in report['per_class'].values()]
    ap = [x['average_precision'] for x in report['per_class'].values()]
    report['macro_auroc'] = float(np.mean(ranks)) if all(v is not None for v in ranks) else None
    report['macro_auprc_average_precision'] = float(np.mean(ap)) if all(v is not None for v in ap) else None
    report['threshold'] = .5
    report['undefined_policy'] = 'Ranking requires both outcomes; undefined is null. Precision/recall/F1 undefined denominators use zero, with predicted-positive support reported.'
    for j, value in enumerate(report['per_class'].values()):
        value['predicted_positive'] = int(pred[:, j].sum())
        value['binary_accuracy'] = float((pred[:, j] == np.asarray(y)[:, j]).mean())
    return report


def select_backbone(order, reports):
    if set(reports) != set(order):
        return None
    scores = [reports[n]['patient']['macro_auroc'] for n in order]
    if any(s is None or not np.isfinite(s) for s in scores):
        return None
    return order[int(np.argmax(scores))]
