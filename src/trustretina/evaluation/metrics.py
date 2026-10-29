"""Equal-shaped two-dimensional multilabel metrics; no fitting or argmax."""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from trustretina.config import CLASSES


def binary_targets(values):
    y = np.asarray(values)
    if y.ndim != 2 or y.shape[1] != 8 or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("Targets must be nonempty binary (n,8) arrays")
    return y.astype(np.int64)


def probabilities(values, shape=None):
    p = np.asarray(values, dtype=float)
    if (
        p.ndim != 2
        or p.shape[1] != 8
        or len(p) == 0
        or (shape is not None and p.shape != shape)
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError(
            "Probabilities must be finite, equal-shaped (n,8) values in [0,1]"
        )
    return p


def multilabel_metrics(targets, scores, thresholds=0.5):
    y = binary_targets(targets)
    p = probabilities(scores, y.shape)
    t = np.asarray(thresholds, dtype=float)
    if (
        t.shape not in ((), (8,))
        or not np.isfinite(t).all()
        or ((t < 0) | (t > 1)).any()
    ):
        raise ValueError("Require one or eight finite declared thresholds")
    pred = (p >= t).astype(int)
    report = {
        "class_order": CLASSES,
        "n": len(y),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "macro_precision": float(
            precision_score(y, pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "brier": float(np.mean((p - y) ** 2)),
        "per_class": {},
    }
    f1 = f1_score(y, pred, average=None, zero_division=0)
    precision = precision_score(y, pred, average=None, zero_division=0)
    recall = recall_score(y, pred, average=None, zero_division=0)
    for j, c in enumerate(CLASSES):
        pos = int(y[:, j].sum())
        neg = len(y) - pos
        both = pos > 0 and neg > 0
        report["per_class"][c] = {
            "positive": pos,
            "negative": neg,
            "f1": float(f1[j]),
            "precision": float(precision[j]),
            "recall": float(recall[j]),
            "auroc": float(roc_auc_score(y[:, j], p[:, j])) if both else None,
            "average_precision": (
                float(average_precision_score(y[:, j], p[:, j])) if both else None
            ),
            "ranking_undefined_reason": (
                None if both else "Both outcomes required by reporting protocol"
            ),
            "brier": float(np.mean((p[:, j] - y[:, j]) ** 2)),
        }
    return report


def aggregate_patient_mean(patient_ids, eyes, targets, scores, component_ids=None):
    y = binary_targets(targets)
    p = probabilities(scores, y.shape)
    if (
        len(patient_ids) != len(y)
        or len(eyes) != len(y)
        or (component_ids is not None and len(component_ids) != len(y))
    ):
        raise ValueError("Row metadata length mismatch")
    groups = {}
    for i, (pid, eye) in enumerate(zip(patient_ids, eyes)):
        if not isinstance(pid, str) or not pid or eye not in ("left", "right"):
            raise ValueError("Require patient IDs and declared eye sides")
        rows = groups.setdefault(pid, {})
        if eye in rows:
            raise ValueError("Repeated eye or patient across partitions")
        rows[eye] = i
    ids = []
    labels = []
    preds = []
    components = []
    for pid, rows in sorted(groups.items()):
        if set(rows) != {"left", "right"}:
            raise ValueError("Both eyes required for patient aggregation")
        a, b = rows["left"], rows["right"]
        if not np.array_equal(y[a], y[b]):
            raise ValueError("Patient-derived labels disagree between eyes")
        if component_ids is not None and component_ids[a] != component_ids[b]:
            raise ValueError("Patient spans duplicate components")
        ids.append(pid)
        labels.append(y[a])
        preds.append((p[a] + p[b]) / 2)
        components.append(component_ids[a] if component_ids is not None else pid)
    return {
        "patient_ids": ids,
        "labels": np.stack(labels),
        "probabilities": np.stack(preds),
        "component_ids": components,
        "aggregation": "mean of left/right sigmoid probabilities before applying thresholds; patient-derived labels",
    }


def component_bootstrap_f1(
    targets, scores, component_ids, *, replicates=1000, seed=2026, confidence=0.95
):
    """Cluster bootstrap of patient rows; duplicate-connected patients resample together.

    Descriptive percentile intervals, not simultaneous or small-sample guarantees.
    Undefined per-class rankings remain undefined; no seed search.
    """
    y = binary_targets(targets)
    p = probabilities(scores, y.shape)
    if len(component_ids) != len(y) or replicates < 20 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap contract")
    unique = sorted(set(component_ids))
    if len(unique) < 2:
        raise ValueError("At least two independent patient components required")
    rows = {g: np.flatnonzero(np.asarray(component_ids) == g) for g in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        idx = np.concatenate(
            [rows[unique[i]] for i in rng.integers(0, len(unique), len(unique))]
        )
        values.append(f1_score(y[idx], p[idx] >= 0.5, average="macro", zero_division=0))
    alpha = (1 - confidence) / 2
    return {
        "macro_f1_percentile_interval": np.quantile(
            values, [alpha, 1 - alpha]
        ).tolist(),
        "confidence": confidence,
        "replicates": replicates,
        "seed": seed,
        "resampling_unit": "duplicate-connected patient component",
        "limitation": "Descriptive; rare-class uncertainty and multiplicity require separate reporting",
    }
