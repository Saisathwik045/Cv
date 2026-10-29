"""Functional future training path. No fitting occurs without explicit protocol approval."""

from pathlib import Path
import math, json
import torch
from .protocol import ProtocolApproval
from .checkpoint import save_checkpoint
from trustretina.config import BACKBONES


def training_loss(logits, targets, pos_weight=None):
    if (
        logits.ndim != 2
        or len(logits) == 0
        or logits.shape[1] != 8
        or logits.shape != targets.shape
        or not torch.isfinite(logits).all()
        or not torch.isin(
            targets, torch.tensor([0.0, 1.0], device=targets.device)
        ).all()
    ):
        raise ValueError("BCE requires finite equal-shaped binary (n,8) targets/logits")
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets.float(), pos_weight=pos_weight
    )


def train_epoch(
    model,
    batches,
    optimizer,
    *,
    approval=None,
    allow_fitting=False,
    device="cpu",
    pos_weight=None,
):
    (approval or ProtocolApproval()).require_fit(
        allow_fitting=allow_fitting, role="train"
    )
    model.train()
    total = 0.0
    count = 0
    steps = 0
    for batch in batches:
        if (
            batch.get("role") != "train"
            or batch.get("manifest_sha256")
            != dict(approval.partition_checksums)["train"]
        ):
            raise ValueError(
                "Only approved immutable training patients may update weights"
            )
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["images"].to(device))
        loss = training_loss(logits, batch["targets"].to(device), pos_weight)
        loss.backward()
        optimizer.step()
        n = len(logits)
        total += float(loss.detach()) * n
        count += n
        steps += 1
    if not count:
        raise ValueError("Empty training loader")
    return {"loss": total / count, "steps": steps}


def select_checkpoint(reports, *, role="validation"):
    if role != "validation" or any(r.get("role") != "validation" for r in reports):
        raise ValueError("Checkpoint selection uses validation only")
    if not reports or any(
        not math.isfinite(r["macro_f1"]) or not 0 <= r["macro_f1"] <= 1 for r in reports
    ):
        raise ValueError("Require finite validation reports")
    # Highest validation macro F1 at fixed .5; ties keep earliest epoch.
    return min(reports, key=lambda r: (-r["macro_f1"], r["epoch"]))


def select_backbone(reports):
    if set(reports) != set(BACKBONES) or any(
        r.get("role") != "validation" for r in reports.values()
    ):
        raise ValueError("Require five validation-only backbone reports")
    if any(
        not math.isfinite(r["macro_f1"]) or not 0 <= r["macro_f1"] <= 1
        for r in reports.values()
    ):
        raise ValueError("Invalid validation selection score")
    return min(
        BACKBONES, key=lambda name: (-reports[name]["macro_f1"], BACKBONES.index(name))
    )


def run_training(
    model,
    train_batches,
    validation_callback,
    optimizer,
    *,
    epochs,
    output,
    provenance,
    approval=None,
    allow_fitting=False,
    scheduler=None,
    generator=None,
    start_epoch=0,
    global_step=0,
    best_score=None,
    device="cpu",
):
    (approval or ProtocolApproval()).require_fit(
        allow_fitting=allow_fitting, role="train"
    )
    if (
        not 0 <= start_epoch < epochs
        or provenance["split_sha256"] != approval.split_sha256
    ):
        raise ValueError("Invalid epoch budget or approved split provenance")
    output = Path(output)
    if any(
        (output / f"epoch-{e:03d}.json").exists() for e in range(start_epoch, epochs)
    ):
        raise ValueError("Refuse to overwrite epoch provenance before any fitting")
    history = []
    for epoch in range(start_epoch, epochs):
        train = train_epoch(
            model,
            train_batches,
            optimizer,
            approval=approval,
            allow_fitting=allow_fitting,
            device=device,
        )
        global_step += train["steps"]
        validation = validation_callback(model)
        if (
            validation.get("role") != "validation"
            or validation.get("manifest_sha256")
            != dict(approval.partition_checksums)["validation"]
            or validation.get("thresholds") != 0.5
        ):
            raise ValueError("Selection callback must be validation-only")
        score = validation["macro_f1"]
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid validation score")
        if scheduler is not None:
            scheduler.step()
        improved = best_score is None or score > best_score
        if improved:
            best_score = score
        args = dict(
            completed_epoch=epoch,
            global_step=global_step,
            provenance=provenance,
            scheduler=scheduler,
            generator=generator,
            best_score=best_score,
        )
        save_checkpoint(output / "last.pt", model, optimizer, **args)
        if improved:
            save_checkpoint(output / "best.pt", model, optimizer, **args)
        history.append(
            {"epoch": epoch, "train_loss": train["loss"], "validation_macro_f1": score}
        )
        history_path = output / f"epoch-{epoch:03d}.json"
        if history_path.exists():
            raise ValueError(
                "Refuse to overwrite epoch provenance; choose a fresh resume output directory"
            )
        history_path.write_text(
            json.dumps({"provenance": provenance, **history[-1]}, sort_keys=True)
        )
    return history
