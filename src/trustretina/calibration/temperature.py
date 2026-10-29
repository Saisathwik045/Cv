"""Explicit temperature application and gated future fitting; no import-time fitting."""

from dataclasses import dataclass
import numpy as np
import torch
from trustretina.training.protocol import ProtocolApproval, ProtocolNotReady
from trustretina.evaluation.metrics import binary_targets


@dataclass(frozen=True)
class TemperatureCalibration:
    temperatures: tuple = (1.0,)
    source_role: str = "identity_unfitted"

    def __post_init__(self):
        values = np.asarray(self.temperatures)
        if (
            values.shape not in ((1,), (8,))
            or not np.isfinite(values).all()
            or (values <= 0).any()
        ):
            raise ValueError("Require one or eight finite positive temperatures")

    def probabilities(self, logits):
        x = np.asarray(logits, dtype=float)
        if x.ndim != 2 or x.shape[1] != 8 or not np.isfinite(x).all():
            raise ValueError("Require finite (n,8) logits")
        # Numerically stable sigmoid, preserving all eight independent labels.
        z = x / np.asarray(self.temperatures)
        return np.exp(-np.logaddexp(0.0, -z))


def fit_temperature(
    logits,
    targets,
    *,
    role,
    approval=None,
    allow_fitting=False,
    per_class=False,
    patient_ids=None,
    component_ids=None,
    manifest_sha256=None,
    min_positive=20,
    min_negative=20,
):
    (approval or ProtocolApproval()).require_fit(allow_fitting=allow_fitting, role=role)
    if role != "calibration_fit":
        raise ProtocolNotReady("Temperature fitting requires calibration_fit patients")
    y = binary_targets(targets)
    x = np.asarray(logits, dtype=float)
    if x.shape != y.shape or not np.isfinite(x).all():
        raise ValueError("Require equal-shaped finite logits and labels")
    if manifest_sha256 != dict(approval.partition_checksums)[role]:
        raise ProtocolNotReady(
            "Calibration predictions do not bind to approved partition"
        )
    if (
        patient_ids is None
        or component_ids is None
        or len(patient_ids) != len(y)
        or len(set(patient_ids)) != len(y)
        or len(component_ids) != len(y)
    ):
        raise ValueError(
            "Calibration requires one row per patient with duplicate-component metadata"
        )
    groups = {}
    for i, group in enumerate(component_ids):
        groups.setdefault(group, []).append(i)
    independent = []
    for rows in groups.values():
        if not np.all(y[rows] == y[rows[0]]):
            raise ProtocolNotReady(
                "Mixed-label duplicate group requires reviewed calibration objective; do not resolve labels by voting"
            )
        independent.append(y[rows[0]])
    support = np.stack(independent)
    positives = support.sum(0)
    negatives = len(support) - positives
    if (
        min_positive < 20
        or min_negative < 20
        or (positives < min_positive).any()
        or (negatives < min_negative).any()
    ):
        raise ProtocolNotReady(
            "Insufficient independent per-class calibration support; floors are safeguards, not power guarantees"
        )
    # Equal connected-group mass prevents repeated copies overweighting calibration.
    row_weights = torch.tensor(
        [1 / len(groups[g]) for g in component_ids], dtype=torch.float64
    )
    parameter = torch.zeros(
        8 if per_class else 1, dtype=torch.float64, requires_grad=True
    )
    tx = torch.tensor(x)
    ty = torch.tensor(y, dtype=torch.float64)
    optimizer = torch.optim.LBFGS(
        [parameter], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                tx / parameter.exp(), ty, reduction="none"
            ).mean(1)
            * row_weights
        ).sum() / row_weights.sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return TemperatureCalibration(tuple(parameter.detach().exp().tolist()), role)


def patient_logits(mean_eye_probabilities, *, epsilon=1e-6):
    from trustretina.evaluation.metrics import probabilities

    if epsilon != 1e-6:
        raise ValueError("Protocol fixes clipping at 1e-6")
    p = np.clip(probabilities(mean_eye_probabilities), epsilon, 1 - epsilon)
    return np.log(p) - np.log1p(-p)
