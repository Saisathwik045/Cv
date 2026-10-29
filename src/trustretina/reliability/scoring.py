"""Inference-only MC dropout, documented technical quality proxies and fixed triage policy."""

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from trustretina.training.protocol import ProtocolApproval, ProtocolNotReady


def mc_dropout(model, images, *, passes=20):
    if passes < 2:
        raise ValueError("At least two passes required")
    flags = {m: m.training for m in model.modules()}
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.modules.dropout._DropoutNd):
            m.train()
    try:
        with torch.inference_mode():
            samples = torch.stack([torch.sigmoid(model(images)) for _ in range(passes)])
    finally:
        for m, flag in flags.items():
            m.training = flag
    if samples.ndim != 3 or samples.shape[-1] != 8:
        raise ValueError("Require (passes,n,8) independent probabilities")
    return {
        "mean": samples.mean(0),
        "variance": samples.var(0, unbiased=False),
        "passes": passes,
    }


def quality_proxies(rgb):
    x = np.asarray(rgb, dtype=float)
    if (
        x.ndim != 3
        or min(x.shape[:2]) < 2
        or x.shape[2] != 3
        or not np.isfinite(x).all()
        or x.min() < 0
        or x.max() > 255
    ):
        raise ValueError("Require RGB [0,255]")
    gray = x.mean(2) / 255.0
    foreground = gray > 8 / 255.0
    if foreground.sum() < 4:
        return {
            "foreground_fraction": float(foreground.mean()),
            "contrast": 0.0,
            "gradient_energy": 0.0,
        }
    # Technical proxies only; no diagnostic/clinical quality claim.
    return {
        "foreground_fraction": float(foreground.mean()),
        "contrast": float(gray[foreground].std()),
        "gradient_energy": float(
            np.mean(np.diff(gray, axis=0) ** 2) + np.mean(np.diff(gray, axis=1) ** 2)
        ),
    }


@dataclass(frozen=True)
class ReliabilityPolicy:
    minimum_confidence: float = 0.95
    maximum_variance: float = 0.01
    minimum_quality: float = 0.5
    supported_classes: tuple = (False,) * 8
    fitted: bool = False

    def __post_init__(self):
        if (
            len(self.supported_classes) != 8
            or not 0 <= self.minimum_confidence <= 1
            or not 0 <= self.maximum_variance <= 0.25
            or not 0 <= self.minimum_quality <= 1
        ):
            raise ValueError("Invalid fixed reliability policy")

    def decide(self, probabilities, variance, quality):
        p = np.asarray(probabilities)
        v = np.asarray(variance)
        q = np.asarray(quality)
        if (
            p.ndim != 2
            or p.shape[1] != 8
            or p.shape != v.shape
            or q.shape != (len(p),)
            or not np.isfinite(p).all()
            or not np.isfinite(v).all()
            or not np.isfinite(q).all()
            or (p < 0).any()
            or (p > 1).any()
            or (v < 0).any()
            or (q < 0).any()
            or (q > 1).any()
        ):
            raise ValueError("Invalid reliability arrays")
        confidence = np.maximum(p, 1 - p)
        allowed = self.fitted and all(self.supported_classes)
        accept = (
            allowed
            & (confidence.min(1) >= self.minimum_confidence)
            & (v.max(1) <= self.maximum_variance)
            & (q >= self.minimum_quality)
        )
        return np.where(accept, "ACCEPT", "REVIEW REQUIRED")


def fit_reliability(*, role, approval=None, allow_fitting=False, **evidence):
    (approval or ProtocolApproval()).require_fit(allow_fitting=allow_fitting, role=role)
    if role != "calibration_policy":
        raise ProtocolNotReady(
            "Reliability policy fitting requires disjoint calibration_policy patients"
        )
    # Deliberately no generic risk optimizer: its loss, risk budget, search family,
    # multiplicity and minimum support must be prespecified and reviewed first.
    raise ProtocolNotReady(
        "Risk objective/confidence bounds and class support protocol not approved; no policy fitted"
    )
