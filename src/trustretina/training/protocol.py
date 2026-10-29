"""Explicit fitting/final-evaluation authorization. Historical bundles are not approved."""

from dataclasses import dataclass
import hashlib, json, math
from pathlib import Path


class ProtocolNotReady(PermissionError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class ProtocolApproval:
    training_ready: bool = False
    protocol_id: str = "unapproved"
    split_sha256: str = ""
    calibration_ready: bool = False
    risk_budget: float | None = None
    partition_checksums: tuple = ()

    def require_fit(self, *, allow_fitting=False, role="train"):
        if not allow_fitting or not self.training_ready:
            raise ProtocolNotReady(
                "Fitting disabled: unresolved duplicate/split/calibration protocol; explicit approved authorization required"
            )
        if role not in dict(self.partition_checksums) or len(self.split_sha256) != 64:
            raise ProtocolNotReady("Approved immutable split/role binding required")
        if role not in ("train", "validation", "calibration_fit", "calibration_policy"):
            raise ProtocolNotReady("Fitting cannot access final evaluation")
        if role.startswith("calibration") and not self.calibration_ready:
            raise ProtocolNotReady("Calibration/reliability support not approved")


def load_approval(path, *, expected_sha256):
    if sha256(path) != expected_sha256:
        raise ProtocolNotReady("Protocol approval checksum mismatch")
    record = json.loads(Path(path).read_text())
    required = (
        "training_ready",
        "duplicate_review_complete",
        "split_approved",
        "final_evaluation_locked",
        "calibration_plan_approved",
        "power_limitations_reviewed",
    )
    if any(record.get(k) is not True for k in required):
        raise ProtocolNotReady("Protocol is not approved for fitting")
    risk = record.get("risk_budget")
    if type(risk) not in (int, float) or not math.isfinite(risk) or not 0 < risk < 1:
        raise ProtocolNotReady(
            "Prespecified reliability risk budget required; no clinical default"
        )
    if (
        not isinstance(record.get("split_sha256"), str)
        or len(record["split_sha256"]) != 64
    ):
        raise ProtocolNotReady("Invalid approved split checksum")
    partitions = record.get("partition_checksums", {})
    if set(partitions) != {
        "train",
        "validation",
        "calibration_fit",
        "calibration_policy",
        "final_evaluation",
    } or any(not isinstance(v, str) or len(v) != 64 for v in partitions.values()):
        raise ProtocolNotReady("Missing immutable partition checksums")
    return ProtocolApproval(
        True,
        record["protocol_id"],
        record["split_sha256"],
        record.get("calibration_ready") is True,
        record.get("risk_budget"),
        tuple(sorted(partitions.items())),
    )


def require_final_freeze(path, *, expected_sha256):
    if not path or not expected_sha256 or sha256(path) != expected_sha256:
        raise ProtocolNotReady(
            "Final evaluation requires trusted frozen artifact manifest"
        )
    record = json.loads(Path(path).read_text())
    if (
        record.get("frozen") is not True
        or record.get("final_evaluation_release") is not True
    ):
        raise ProtocolNotReady("Final evaluation has not been released")
    required = {
        "checkpoint",
        "disease_thresholds",
        "calibrator",
        "reliability_policy",
        "preprocessing",
        "split",
    }
    if set(record.get("artifacts", {})) != required:
        raise ProtocolNotReady("Incomplete frozen pipeline")
    for item in record["artifacts"].values():
        target = (Path(path).parent / item["path"]).resolve()
        if sha256(target) != item["sha256"]:
            raise ProtocolNotReady("Frozen artifact changed")
    return record
