"""Atomic epoch-boundary checkpoints with safe tensor-only loading and RNG provenance."""

import hashlib, os, random, json, platform
from pathlib import Path
import numpy as np
import torch


def capture_rng(generator=None):
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": state[0],
            "keys": torch.tensor(state[1].astype(np.int64)),
            "position": state[2],
            "has_gauss": state[3],
            "gauss": state[4],
        },
        "torch": torch.get_rng_state(),
        "loader_generator": generator.get_state() if generator is not None else None,
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state, generator=None):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state(
        (
            n["name"],
            n["keys"].numpy().astype(np.uint32),
            n["position"],
            n["has_gauss"],
            n["gauss"],
        )
    )
    torch.set_rng_state(state["torch"])
    if state["loader_generator"] is not None:
        if generator is None:
            raise ValueError("Checkpoint requires explicit data-loader generator")
        generator.set_state(state["loader_generator"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG checkpoint cannot resume as a CPU run")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path,
    model,
    optimizer,
    *,
    completed_epoch,
    global_step,
    provenance,
    scheduler=None,
    scaler=None,
    generator=None,
    best_score=None,
):
    if completed_epoch < 0 or global_step < 0:
        raise ValueError("Only completed epoch-boundary checkpoints supported")
    required = {
        "backbone",
        "split_sha256",
        "config_sha256",
        "preprocessing_sha256",
        "seed",
        "data_kind",
    }
    if not required <= set(provenance):
        raise ValueError("Missing checkpoint provenance")
    record = {
        "schema": 1,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
        },
        "boundary": "completed_epoch",
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "best_score": best_score,
        "provenance": provenance,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": capture_rng(generator),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(record, handle)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(path)
    marker = path.with_suffix(path.suffix + ".sha256")
    temp_marker = marker.with_suffix(marker.suffix + ".tmp")
    temp_marker.write_text(digest)
    temp_marker.replace(marker)
    return digest


def load_checkpoint(
    path,
    model,
    optimizer,
    *,
    expected_sha256,
    expected_provenance,
    scheduler=None,
    scaler=None,
    generator=None,
):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint checksum mismatch")
    record = torch.load(path, map_location="cpu", weights_only=True)
    if (
        record.get("schema") != 1
        or record.get("boundary") != "completed_epoch"
        or record.get("provenance") != expected_provenance
    ):
        raise ValueError("Checkpoint protocol/data/config provenance mismatch")
    if record.get("environment") != {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
    }:
        raise ValueError(
            "Resume environment changed; deterministic continuation not established"
        )
    if record["rng"]["cuda"] and not torch.cuda.is_available():
        raise ValueError("CUDA RNG state cannot resume as CPU")
    if (record["scheduler"] is None) != (scheduler is None) or (
        record["scaler"] is None
    ) != (scaler is None):
        raise ValueError("Scheduler/scaler resume contract changed")
    if record["rng"]["loader_generator"] is not None and generator is None:
        raise ValueError("Missing resumable loader generator")
    model.load_state_dict(record["model"], strict=True)
    optimizer.load_state_dict(record["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(record["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(record["scaler"])
    restore_rng(record["rng"], generator)
    return {
        "next_epoch": record["completed_epoch"] + 1,
        "global_step": record["global_step"],
        "best_score": record["best_score"],
    }
