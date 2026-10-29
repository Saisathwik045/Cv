"""Prospective four-role development protocol; no historical split approval by reuse."""

from pathlib import Path
import json
from trustretina.config import CLASSES, PARTITIONS
from .splits import allocate_components, ALGORITHM
from .duplicates import components
from .validation import DataValidationError, canonical_hash
from .review_policy import support_summary, assert_partition_separation

PROTOCOL = "development-shared-calibration-v3"


def development_support(patients, ids, groups):
    summary = support_summary(patients, ids, groups)
    by_id = {p["patient_id"]: p for p in patients}
    chosen = set(ids)
    selected = [[by_id[pid] for pid in group if pid in chosen] for group in groups]
    selected = [g for g in selected if g]
    summary["component_size_histogram"] = {
        str(k): v for k, v in summary["component_size_histogram"].items()
    }
    summary["positive_groups"] = [
        sum(any(p["labels"][j] for p in g) for g in selected) for j in range(8)
    ]
    summary["negative_groups"] = [
        sum(any(not p["labels"][j] for p in g) for g in selected) for j in range(8)
    ]
    summary["mixed_label_groups"] = [
        sum(
            any(p["labels"][j] for p in g) and any(not p["labels"][j] for p in g)
            for g in selected
        )
        for j in range(8)
    ]
    return summary


def development_assignment(inventory, verified_links, config):
    """Only evidence-supported links are passed, never the entire dHash graph."""
    patients = inventory["patients"]
    groups = components(patients, verified_links)
    partitions, repairs = allocate_components(patients, groups, config)
    assigned = {
        role: [p["patient_id"] for p in partitions[k]]
        for k, role in enumerate(PARTITIONS)
    }
    assert_partition_separation(assigned, groups)
    summary = {
        role: development_support(patients, ids, groups)
        for role, ids in assigned.items()
    }
    return {
        "schema_version": 3,
        "class_order": CLASSES,
        "protocol": PROTOCOL,
        "algorithm": ALGORITHM,
        "dataset_fingerprint": inventory["fingerprint"],
        "seed": config["seed"],
        "settings": config["split"],
        "partitions": assigned,
        "support": summary,
        "support_repairs": repairs,
        "patient_groups": groups,
        "partition_approved": False,
        "pilot_ready": False,
        "full_training_ready": False,
        "calibration_design": "one shared temperature on the entire calibration partition; no calibration-policy fit; unsupported clinical triage remains REVIEW REQUIRED",
    }


def audit_development(
    manifest,
    inventory,
    verified_links,
    material_cases,
    *,
    review_complete,
    controls_passed,
    raw_verified,
):
    if (
        manifest.get("schema_version") != 3
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("algorithm") != ALGORITHM
        or set(manifest.get("partitions", {})) != set(PARTITIONS)
    ):
        raise DataValidationError(
            "Invalid four-role development manifest schema/method"
        )
    groups = components(inventory["patients"], verified_links)
    if manifest["patient_groups"] != groups:
        raise DataValidationError("Verified-copy component mutation")
    assert_partition_separation(manifest["partitions"], groups)
    owner = {pid: role for role, ids in manifest["partitions"].items() for pid in ids}
    expected = {p["patient_id"] for p in inventory["patients"]}
    if (
        set(owner) != expected
        or manifest["dataset_fingerprint"] != inventory["fingerprint"]
        or manifest["class_order"] != CLASSES
    ):
        raise DataValidationError("Dataset/partition/label-order mismatch")
    recomputed = {
        role: development_support(inventory["patients"], ids, groups)
        for role, ids in manifest["partitions"].items()
    }
    if manifest["support"] != recomputed:
        raise DataValidationError("Support summary was altered")
    conflicts = []
    for pair in verified_links:
        if pair["conflicting_patient_labels"]:
            conflicts.append(pair)
    crossings = []
    for case in material_cases:
        a, b = (case[k]["patient_id"] for k in ("a", "b"))
        if a not in owner or b not in owner:
            raise DataValidationError("Unknown unresolved candidate endpoint")
        if owner[a] != owner[b]:
            crossings.append(case)
    support_ok = all(
        min(s["positive"]) >= manifest["settings"]["minimum_positive_patients"]
        and min(s["negative"]) >= manifest["settings"]["minimum_negative_patients"]
        and min(s["positive_groups"])
        >= manifest["settings"]["minimum_positive_patients"]
        and min(s["negative_groups"])
        >= manifest["settings"]["minimum_negative_patients"]
        for s in recomputed.values()
    )
    calibrated_groups = recomputed["calibration"]["components"]
    reasons = []
    if not review_complete:
        reasons.append("bounded_review_incomplete")
    if not controls_passed:
        reasons.append("photographic_controls_failed")
    if not raw_verified:
        reasons.append("raw_input_validation_incomplete")
    if not support_ok:
        reasons.append("class_support_floor_failed")
    if calibrated_groups < 50:
        reasons.append("insufficient_groups_for_shared_calibration")
    if conflicts:
        reasons.append("evidence_supported_copy_annotation_conflicts_require_review")
    if crossings:
        reasons.append("material_unresolved_copy_candidates_cross_partitions")
    return {
        "checks_passed": not reasons,
        "pilot_ready": not reasons,
        "full_training_ready": False,
        "blockers": reasons,
        "material_cases": len(material_cases),
        "material_cross_partition_cases": len(crossings),
        "crossing_cases": crossings,
        "verified_link_annotation_conflicts": len(conflicts),
        "verified_link_conflicts": conflicts,
        "support": recomputed,
        "raw_verified": bool(raw_verified),
        "controls_passed": bool(controls_passed),
        "review_complete": bool(review_complete),
    }


def write_diagnostic(directory, manifest, audit):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in [
        ("candidate-assignment.json", manifest),
        ("readiness.json", audit),
    ]:
        path = directory / name
        path.write_text(json.dumps(value, sort_keys=True, indent=2))
        path.with_suffix(".sha256").write_text(
            __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        )
    # A ready decision is still not a publication: release must bind the real
    # review/source/config checksums and scoped pilot authorization separately.
    return directory


def load_development_config(path):
    import tomllib, math
    from trustretina.config import BACKBONES

    config = tomllib.loads(Path(path).read_text())
    if (
        config.get("schema_version") != 3
        or config.get("protocol") != PROTOCOL
        or config.get("class_order") != CLASSES
        or config.get("backbones") != BACKBONES
        or config.get("seed") != 2026
    ):
        raise ValueError(
            "Require frozen v3 protocol, class/backbone order and documented seed 2026"
        )
    split = config["split"]
    fractions = split["fractions"]
    if (
        len(fractions) != 4
        or any(not math.isfinite(f) or not 0 < f < 1 for f in fractions)
        or not math.isclose(sum(fractions), 1, abs_tol=1e-9)
        or split["method"] != ALGORITHM
    ):
        raise ValueError("Invalid four-role stratification contract")
    if any(
        type(split[k]) is not int or split[k] < 1
        for k in ("minimum_positive_patients", "minimum_negative_patients")
    ):
        raise ValueError("Positive/negative support floors must be positive integers")
    cal = config["calibration"]
    if (
        cal["method"] != "shared_patient_temperature"
        or cal["role"] != "calibration"
        or cal["class_specific_fitting"] is not False
        or cal["policy_fitting"] is not False
        or cal["minimum_independent_groups"] != 50
    ):
        raise ValueError(
            "Require single shared calibration, no per-class or policy fitting"
        )
    pilot = config["pilot"]
    if (
        pilot["backbone"] != "efficientnet_b0"
        or type(pilot["maximum_epochs"]) is not int
        or not 1 <= pilot["maximum_epochs"] <= 2
        or not 1 <= pilot["maximum_training_patients"] <= 128
        or not 1 <= pilot["maximum_validation_patients"] <= 64
        or pilot["uses_final_evaluation"] is not False
        or pilot["full_five_model_training_authorized"] is not False
        or pilot["requires_data_checks_passed"] is not True
    ):
        raise ValueError("Pilot exceeds the one bounded execution authorization")
    selection = config["selection"]
    if (
        selection["role"] != "validation"
        or selection["metric"] != "patient_macro_f1"
        or selection["threshold"] != 0.5
        or selection["final_evaluation_for_selection"] is not False
    ):
        raise ValueError("Selection must use validation only at declared .5 threshold")
    return config


def freeze_for_pilot(
    directory, manifest, audit, inventory, *, config_sha256, evidence_sha256
):
    """Only a successful real-data audit can create a new immutable four-role release.

    This never authorizes full training or clinical calibration/reliability claims.
    """
    import hashlib

    if (
        audit.get("checks_passed") is not True
        or audit.get("pilot_ready") is not True
        or audit.get("blockers")
        or audit.get("full_training_ready") is not False
    ):
        raise DataValidationError(
            "Dataset/protocol checks block split freeze and GPU pilot"
        )
    if len(config_sha256) != 64 or len(evidence_sha256) != 64:
        raise DataValidationError("Trusted config/evidence checksums required")
    if inventory.get('data_kind')!='original_odir':raise DataValidationError('Synthetic/unknown data cannot be frozen as real development readiness')
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError("Never overwrite a frozen manifest version")
    destination.mkdir(parents=True)
    by_id = {p["patient_id"]: p for p in inventory["patients"]}
    components_by_id = {
        pid: canonical_hash(g) for g in manifest["patient_groups"] for pid in g
    }
    hashes = {}
    for role, ids in manifest["partitions"].items():
        rows = []
        for pid in ids:
            p = by_id[pid]
            for eye, r in sorted(p["eyes"].items()):
                rows.append(
                    {
                        "patient_id": pid,
                        "eye": eye,
                        "labels": p["labels"],
                        "component_id": components_by_id[pid],
                        "path": r["path"],
                        "raw_sha256": r["sha256"],
                    }
                )
        payload = {
            "schema": "phase2_eye_rows_v1",
            "protocol": PROTOCOL,
            "role": role,
            "rows": rows,
            "dataset_fingerprint": inventory["fingerprint"],
        }
        file = destination / f"{role}.json"
        file.write_text(json.dumps(payload, sort_keys=True, indent=2))
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    summary = {
        **manifest,
        "partition_approved": True,
        "pilot_ready": True,
        "full_training_ready": False,
        "config_sha256": config_sha256,
        "evidence_sha256": evidence_sha256,
        "artifact_checksums": hashes,
    }
    file = destination / "manifest.json"
    file.write_text(json.dumps(summary, sort_keys=True, indent=2))
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    file.with_suffix(".sha256").write_text(digest)
    return summary, digest
