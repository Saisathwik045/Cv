"""Inference-only multilabel evaluation; final results require a frozen release."""

import numpy as np
import torch
from .metrics import multilabel_metrics, aggregate_patient_mean
from trustretina.training.protocol import require_final_freeze, ProtocolNotReady


def evaluate(
    model,
    batches,
    *,
    role="validation",
    device="cpu",
    thresholds=0.5,
    freeze_path=None,
    freeze_sha256=None,
):
    if role == "final_evaluation":
        require_final_freeze(freeze_path, expected_sha256=freeze_sha256)
        raise ProtocolNotReady(
            "Final pipeline executor awaits approved calibrator/reliability freeze; validation evaluator cannot release final results"
        )
    elif role not in ("validation", "calibration_fit", "calibration_policy"):
        raise ValueError("Unknown evaluation role")
    flags = {module: module.training for module in model.modules()}
    model.eval()
    ys = []
    ps = []
    pids = []
    eyes = []
    components = []
    manifest_digest = None
    try:
        with torch.inference_mode():
            for batch in batches:
                if batch.get("role") != role:
                    raise ValueError("Loader role does not match evaluation access")
                digest = batch.get("manifest_sha256")
                if manifest_digest is not None and digest != manifest_digest:
                    raise ValueError("Evaluation cannot mix partition manifests")
                manifest_digest = digest
                logits = model(batch["images"].to(device))
                if (
                    logits.ndim != 2
                    or logits.shape[1] != 8
                    or logits.shape != batch["targets"].shape
                ):
                    raise ValueError("Require equal-shaped (n,8) logits/targets")
                ys.append(batch["targets"].cpu().numpy())
                ps.append(torch.sigmoid(logits).cpu().numpy())
                pids.extend(batch["patient_ids"])
                eyes.extend(batch["eyes"])
                components.extend(batch["component_ids"])
    finally:
        for module, flag in flags.items():
            module.training = flag
    if not ys:
        raise ValueError("Empty evaluation loader")
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    patient = aggregate_patient_mean(pids, eyes, y, p, components)
    return {
        "macro_f1": multilabel_metrics(
            patient["labels"], patient["probabilities"], thresholds
        )["macro_f1"],
        "role": role,
        "per_image": multilabel_metrics(y, p, thresholds),
        "per_patient": multilabel_metrics(
            patient["labels"], patient["probabilities"], thresholds
        ),
        "patient_predictions": patient,
        "image_probabilities": p,
        "image_targets": y,
        "thresholds": np.asarray(thresholds).tolist(),
        "manifest_sha256": batch.get("manifest_sha256"),
        "primary_unit": "patient",
        "aggregation": patient["aggregation"],
    }
