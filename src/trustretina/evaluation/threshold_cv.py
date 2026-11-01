"""Exploratory validation threshold CV, separate from training-OOF/deployment APIs."""

import numpy as np
from trustretina.config import CLASSES
from trustretina.data.splits import allocate_components
from trustretina.evaluation.metrics import binary_targets, probabilities


def _identities(values, n):
    values = np.asarray(values).astype(str)
    if values.shape != (n,) or any(not x.strip() for x in values):
        raise ValueError("One nonempty identity per patient required")
    return values


def make_group_folds(y, patient_ids, component_ids):
    """Fixed four-fold label-aware allocator. No probabilities accepted."""
    y = binary_targets(y)
    patients = _identities(patient_ids, len(y))
    components = _identities(component_ids, len(y))
    if len(set(patients)) != len(y):
        raise ValueError("Each patient must occur exactly once")
    rows = sorted(
        ({"patient_id": p, "labels": row.tolist()} for p, row in zip(patients, y)),
        key=lambda row: row["patient_id"],
    )
    units = [sorted(patients[components == c].tolist()) for c in sorted(set(components))]
    groups, repairs = allocate_components(rows, units, {
        "seed": 2026,
        "split": {"fractions": [.25] * 4, "minimum_positive_patients": 1,
                  "minimum_negative_patients": 1},
    })
    assignment = {row["patient_id"]: k for k, group in enumerate(groups) for row in group}
    folds = np.asarray([assignment[p] for p in patients], dtype=np.int64)
    validate_folds(y, patients, components, folds)
    return folds, repairs


def validate_folds(y, patient_ids, component_ids, folds):
    y = binary_targets(y)
    patients = _identities(patient_ids, len(y))
    components = _identities(component_ids, len(y))
    folds = np.asarray(folds)
    if len(set(patients)) != len(y) or folds.shape != (len(y),) or not np.isin(folds, range(4)).all():
        raise ValueError("Unique patients and four integer fold assignments required")
    if set(folds.tolist()) != set(range(4)):
        raise ValueError("All four folds required")
    for component in set(components):
        if len(set(folds[components == component])) != 1:
            raise ValueError("Containment component crosses fitting/assessment folds")
    for k in range(4):
        heldout = y[folds == k]
        if np.any(heldout.sum(0) == 0) or np.any(heldout.sum(0) == len(heldout)):
            raise ValueError("Each assessment fold needs both outcomes for every class")


def support(y, component_ids):
    y = binary_targets(y)
    groups = _identities(component_ids, len(y))
    return [{
        "label": label,
        "positive": int(y[:, j].sum()),
        "negative": int(len(y) - y[:, j].sum()),
        "positive_components": len(set(groups[y[:, j] == 1])),
        "negative_components": len(set(groups[y[:, j] == 0])),
    } for j, label in enumerate(CLASSES)]


def _counts(y, decision):
    positive = y == 1
    tp = int(np.sum(positive & decision))
    fp = int(np.sum(~positive & decision))
    fn = int(np.sum(positive & ~decision))
    tn = int(np.sum(~positive & ~decision))
    return tp, fp, fn, tn


def _metrics(tp, fp, fn, tn):
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "positive": tp + fn, "negative": fp + tn,
            "predicted_positive": tp + fp,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "precision_defined": bool(tp + fp), "recall_defined": bool(tp + fn),
            "f1_defined": bool(2 * tp + fp + fn),
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "false_negative_rate": fn / (tp + fn) if tp + fn else None}


def summarize(y, decisions):
    y = binary_targets(y)
    decisions = binary_targets(decisions).astype(bool)
    if y.shape != decisions.shape:
        raise ValueError("Equal-shaped (n,8) decisions required")
    rows = [{"label": label, **_metrics(*_counts(y[:, j], decisions[:, j]))}
            for j, label in enumerate(CLASSES)]
    pooled = _metrics(*(sum(row[k] for row in rows) for k in ("tp", "fp", "fn", "tn")))
    return {"n": len(y), "per_class": rows,
            "macro_f1": float(np.mean([row["f1"] for row in rows])),
            "micro_f1": pooled["f1"], "total_tp": pooled["tp"],
            "total_fp": pooled["fp"], "total_fn": pooled["fn"],
            "subset_exact_match_accuracy": float(np.all(y == decisions, axis=1).mean()),
            "labelwise_binary_accuracy": float((y == decisions).mean()),
            "hamming_loss": float((y != decisions).mean())}


def fit_thresholds(y, p, component_ids):
    """Fixed-grid macro-F1 objective, using only the supplied fitting rows."""
    y = binary_targets(y)
    p = probabilities(p, y.shape)
    records = support(y, component_ids)
    for j, row in enumerate(records):
        row["fallback"] = row["positive_components"] < 10 or row["negative_components"] < 10
        row["fallback_reason"] = "fewer_than_10_supplying_components_for_an_outcome" if row["fallback"] else None
        row["threshold"] = .5
        row["candidates"] = []
        if row["fallback"]:
            continue
        best = None
        for tick in range(1, 20):
            tp, fp, fn, tn = _counts(y[:, j], p[:, j] >= tick / 20)
            numerator, denominator = 2 * tp, 2 * tp + fp + fn
            row["candidates"].append({"threshold": tick / 20, **_metrics(tp, fp, fn, tn)})
            # Integer ratios, then integer grid distance and higher tick.
            if (best is None or numerator * best[1] > best[0] * denominator
                    or (numerator * best[1] == best[0] * denominator
                        and (-abs(tick - 10), tick) > (-abs(best[2] - 10), best[2]))):
                best = numerator, denominator, tick
        row["threshold"] = best[2] / 20
    return np.asarray([row["threshold"] for row in records]), records


def evaluate_folds(y, p, patient_ids, component_ids, folds, *, role, allow_exploratory=False):
    if role != "validation" or allow_exploratory is not True:
        raise PermissionError("Explicit exploratory validation analysis only; no deployment fitting")
    y = binary_targets(y)
    p = probabilities(p, y.shape)
    groups = _identities(component_ids, len(y))
    folds = np.asarray(folds)
    validate_folds(y, patient_ids, groups, folds)
    decisions = np.zeros_like(y, dtype=bool)
    thresholds = []
    reports = []
    for k in range(4):
        fit, assess = folds != k, folds == k
        t, details = fit_thresholds(y[fit], p[fit], groups[fit])
        decisions[assess] = p[assess] >= t
        thresholds.append(t)
        reports.append({"fold": k, "fit_patients": int(fit.sum()),
                        "assessment_patients": int(assess.sum()),
                        "fit_components": len(set(groups[fit])),
                        "assessment_components": len(set(groups[assess])),
                        "group_overlap": len(set(groups[fit]) & set(groups[assess])),
                        "fitting": details, "assessment_support": support(y[assess], groups[assess]),
                        "baseline": summarize(y[assess], p[assess] >= .5),
                        "threshold_cv": summarize(y[assess], decisions[assess])})
    thresholds = np.asarray(thresholds)
    stability = []
    for j, label in enumerate(CLASSES):
        q1, median, q3 = np.quantile(thresholds[:, j], [.25, .5, .75], method="linear")
        stability.append({"label": label, "fold_thresholds": thresholds[:, j].tolist(),
                          "minimum": float(thresholds[:, j].min()),
                          "maximum": float(thresholds[:, j].max()),
                          "median": float(median), "q1": float(q1), "q3": float(q3),
                          "iqr": float(q3 - q1), "distinct_values": len(set(thresholds[:, j])),
                          "fallback_folds": sum(r["fitting"][j]["fallback"] for r in reports)})
    return {"baseline": summarize(y, p >= .5), "threshold_cv": summarize(y, decisions),
            "folds": reports, "stability": stability}, decisions, thresholds
