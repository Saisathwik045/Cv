"""One prospective shared temperature. No per-class calibration or clinical risk claim."""

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch
from trustretina.training.protocol import ProtocolNotReady, sha256
from trustretina.evaluation.metrics import binary_targets
from .temperature import TemperatureCalibration


@dataclass(frozen=True)
class SharedCalibrationPermit:
    calibration_sha256: str = ""
    selected_checkpoint_sha256: str = ""
    explicitly_authorized: bool = False
    data_checks_passed: bool = False

    def require(self, allow_fitting, manifest_sha256, checkpoint_sha256):
        if (
            not allow_fitting
            or not self.explicitly_authorized
            or not self.data_checks_passed
        ):
            raise ProtocolNotReady(
                "Shared calibration fitting not authorized; the execution pilot does not authorize calibration fitting"
            )
        if (
            len(self.calibration_sha256) != 64
            or len(self.selected_checkpoint_sha256) != 64
            or manifest_sha256 != self.calibration_sha256
            or checkpoint_sha256 != self.selected_checkpoint_sha256
        ):
            raise ProtocolNotReady(
                "Calibration partition/validation-selected checkpoint identity mismatch"
            )


def load_shared_permit(path, *, expected_sha256):
    if sha256(path) != expected_sha256:
        raise ProtocolNotReady("Shared calibration permit checksum mismatch")
    r = json.loads(Path(path).read_text())
    if (
        r.get("protocol") != "development-shared-calibration-v3"
        or r.get("selection_role") != "validation"
        or r.get("clinical_risk_claims") is not False
    ):
        raise ProtocolNotReady("Invalid shared development calibration contract")
    return SharedCalibrationPermit(
        r["calibration_sha256"],
        r["selected_checkpoint_sha256"],
        r.get("calibration_fitting_authorized") is True,
        r.get("data_checks_passed") is True,
    )


def fit_shared_development(
    logits,
    targets,
    *,
    patient_ids,
    component_ids,
    manifest_sha256,
    checkpoint_sha256,
    permit=None,
    allow_fitting=False,
):
    (permit or SharedCalibrationPermit()).require(
        allow_fitting, manifest_sha256, checkpoint_sha256
    )
    y = binary_targets(targets)
    x = np.asarray(logits, dtype=float)
    if (
        x.shape != y.shape
        or not np.isfinite(x).all()
        or len(patient_ids) != len(y)
        or len(set(patient_ids)) != len(y)
        or len(component_ids) != len(y)
    ):
        raise ValueError(
            "Require finite patient-level logits/labels and unique patient/component metadata"
        )
    groups = {g: component_ids.count(g) for g in set(component_ids)}
    if len(groups) < 50 or (y.sum(0) == 0).any() or (y.sum(0) == len(y)).any():
        raise ProtocolNotReady(
            "Shared calibration requires >=50 independent groups and both outcomes for every label; no per-class precision claim"
        )
    weights = torch.tensor([1 / groups[g] for g in component_ids], dtype=torch.float64)
    parameter = torch.zeros(1, dtype=torch.float64, requires_grad=True)
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
            * weights
        ).sum() / weights.sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return TemperatureCalibration(
        tuple(parameter.detach().exp().tolist()), "calibration_shared_development_v3"
    )
