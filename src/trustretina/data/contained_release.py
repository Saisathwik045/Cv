"""Bounded schedule completion and explicit uncertain-content containment, version 4."""

import json, hashlib, tomllib
from pathlib import Path
from collections import Counter
from trustretina.config import CLASSES, PARTITIONS, BACKBONES
from .bounded_review import POLICY_V3, material_uncertainty
from .duplicates import components
from .development import development_assignment, development_support
from .review_policy import assert_partition_separation
from .validation import DataValidationError, canonical_hash, file_hash

POLICY = "scheduled-content-containment-v4"
PROTOCOL = "development-content-containment-v4"


def merge_scheduled(selection, catalog, parts):
    expected = {r["review_index"]: r for r in selection["schedule"]}
    rows = {}
    if len(expected) != 1536:
        raise DataValidationError(
            "The fixed schedule must contain 1536 unique comparisons"
        )
    for part in parts:
        for row in part:
            index = row["review_index"]
            if index not in expected or index in rows:
                raise DataValidationError(
                    "Repeated result or candidate expansion outside fixed schedule"
                )
            spec = expected[index]
            if any(
                row[k] != spec[k]
                for k in ("a_index", "b_index", "pair_index", "candidate")
            ):
                raise DataValidationError("Result endpoint/schedule mismatch")
            for key in ("a", "b"):
                if (
                    row[key + "_photo"]["raw_sha256"]
                    != catalog[row[key + "_index"]]["sha256"]
                ):
                    raise DataValidationError("Result raw image identity mismatch")
            if row["independent_evidence"]["policy_version"] != POLICY_V3["version"]:
                raise DataValidationError("Detailed evidence policy changed")
            rows[index] = row
    missing = sorted(set(expected) - set(rows))
    return [
        rows[r["review_index"]]
        for r in selection["schedule"]
        if r["review_index"] in rows
    ], missing


def group_constraints(
    inventory,
    catalog,
    rows,
    *,
    named_cases,
    prior_possible=(),
    supported_false_positive_indices=(0, 1),
):
    by_id = {p["patient_id"]: p for p in inventory["patients"]}
    named = {r["review_index"] for r in named_cases}
    edges = []
    decisions = []
    for row in rows:
        a, b = (catalog[row[k + "_index"]] for k in ("a", "b"))
        e = row["independent_evidence"]
        i = row["review_index"]
        score = (
            row["coarse_intensity_correlation"],
            row["coarse_gradient_correlation"],
        )
        probable = e.get("link") is True
        material = (
            not probable
            and row.get("pair_index") not in supported_false_positive_indices
            and material_uncertainty(
                (a["patient_id"], b["patient_id"]),
                score,
                e,
                prior_possible_copy=i in named or i in prior_possible,
            )
        )
        if probable or material:
            relation = (
                "verified_copy_evidence"
                if probable
                else "unresolved_content_containment"
            )
            reason = (
                e["reason"]
                if probable
                else "Explicitly co-group independent-content review candidate to contain uncertainty; not a confirmed duplicate or clinical identity"
            )
            edges.append(
                {
                    "a": a,
                    "b": b,
                    "review_index": i,
                    "relationship": relation,
                    "reason": reason,
                    "independent_evidence": e,
                    "conflicting_patient_labels": by_id[a["patient_id"]]["labels"]
                    != by_id[b["patient_id"]]["labels"],
                }
            )
            decisions.append(
                {
                    "review_index": i,
                    "decision": (
                        "must_link_supported_copy" if probable else "contain_unresolved"
                    ),
                    "reason": reason,
                }
            )
        else:
            decisions.append(
                {
                    "review_index": i,
                    "decision": "sensitivity_only",
                    "reason": "No supported copy/material trigger in bounded review; this does not prove absence of duplication",
                }
            )
    return edges, decisions


def audit_and_assign(
    inventory,
    edges,
    rows,
    missing,
    config,
    *,
    validation_reused,
    budget_seconds,
    receipts_verified,
    named_review_complete,
):
    manifest = development_assignment(inventory, edges, config)
    groups = manifest["patient_groups"]
    owner = {pid: role for role, ids in manifest["partitions"].items() for pid in ids}
    assert_partition_separation(manifest["partitions"], groups)
    support = {
        role: development_support(inventory["patients"], ids, groups)
        for role, ids in manifest["partitions"].items()
    }
    blockers = []
    if missing or len(rows) != 1536:
        blockers.append("scheduled_review_incomplete")
    if not validation_reused or not receipts_verified:
        blockers.append("verified_source_or_validation_receipt_missing")
    if not named_review_complete:
        blockers.append("named_material_technical_review_incomplete")
    if budget_seconds > 7200:
        blockers.append("additional_cpu_budget_exceeded")
    floors = config["split"]
    if any(
        min(s["positive"]) < floors["minimum_positive_patients"]
        or min(s["negative"]) < floors["minimum_negative_patients"]
        or min(s["positive_groups"]) < floors["minimum_positive_patients"]
        or min(s["negative_groups"]) < floors["minimum_negative_patients"]
        for s in support.values()
    ):
        blockers.append("patient_or_independent_group_support_infeasible")
    if support["calibration"]["components"] < 50:
        blockers.append("shared_calibration_group_support_infeasible")
    if any(owner[e["a"]["patient_id"]] != owner[e["b"]["patient_id"]] for e in edges):
        blockers.append("copy_or_uncertainty_constraint_crossing")
    manifest.update(
        schema_version=4,
        protocol=PROTOCOL,
        duplicate_policy=POLICY,
        support=support,
        pilot_ready=not blockers,
        partition_approved=not blockers,
        full_training_ready=False,
    )
    audit = {
        "schema_version": 4,
        "protocol": PROTOCOL,
        "duplicate_policy": POLICY,
        "checks_passed": not blockers,
        "pilot_ready": not blockers,
        "full_training_ready": False,
        "blockers": blockers,
        "scheduled_completed": len(rows),
        "scheduled_missing": missing,
        "additional_cpu_seconds": budget_seconds,
        "support": support,
        "constraint_counts": dict(Counter(e["relationship"] for e in edges)),
        "patient_components": len(groups),
        "largest_component": max(map(len, groups)),
        "annotation_vector_disagreement_edges": sum(
            e["conflicting_patient_labels"] for e in edges
        ),
        "known_constraint_crossings": 0,
        "annotation_errors_inferred": False,
        "all_possible_duplicates_ruled_out": False,
        "residual_uncertainty": "Bounded 1536-comparison assay only; unselected weak candidates remain sensitivity evidence. No biometric/clinical identity claim.",
    }
    return manifest, audit


def publish_release(output, manifest, audit, inventory, *, provenance):
    if (
        audit["checks_passed"] is not True
        or audit["scheduled_missing"]
        or inventory.get("data_kind") != "original_odir"
    ):
        raise DataValidationError("Real-data release is blocked")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    by_id = {p["patient_id"]: p for p in inventory["patients"]}
    group = {pid: canonical_hash(g) for g in manifest["patient_groups"] for pid in g}
    checksums = {}
    for role, ids in manifest["partitions"].items():
        rows = [
            {
                "patient_id": pid,
                "eye": eye,
                "labels": by_id[pid]["labels"],
                "component_id": group[pid],
                "path": r["path"],
                "raw_sha256": r["sha256"],
            }
            for pid in ids
            for eye, r in sorted(by_id[pid]["eyes"].items())
        ]
        p = output / f"{role}.json"
        p.write_text(
            json.dumps(
                {
                    "schema": "phase2_eye_rows_v1",
                    "protocol": PROTOCOL,
                    "role": role,
                    "rows": rows,
                },
                sort_keys=True,
            )
        )
        checksums[p.name] = file_hash(p)
    manifest = {
        **manifest,
        "provenance": provenance,
        "partition_checksums": dict(checksums),
    }
    for name, value in [("manifest.json", manifest), ("readiness.json", audit)]:
        p = output / name
        p.write_text(json.dumps(value, sort_keys=True, indent=2))
        checksums[name] = file_hash(p)
    (output / "checksums.json").write_text(
        json.dumps(checksums, sort_keys=True, indent=2)
    )
    return manifest


def pilot_spec_from_release(directory, config_sha256):
    directory = Path(directory)
    checks = json.loads((directory / "checksums.json").read_text())
    for name in ("manifest.json", "readiness.json", "train.json", "validation.json"):
        if file_hash(directory / name) != checks[name]:
            raise DataValidationError("Frozen release checksum mismatch before pilot")
    audit = json.loads((directory / "readiness.json").read_text())
    if audit["pilot_ready"] is not True or audit["checks_passed"] is not True:
        raise DataValidationError("No pilot without passed contained-release audit")
    spec = {
        "scope": "one_efficientnet_b0_execution_pilot_v4",
        "data_checks_passed": True,
        "final_evaluation_access": False,
        "epochs": 2,
        "backbone": "efficientnet_b0",
        "initialization": "random_unfitted_execution_only",
        "batch_size": 16,
        "split_sha256": checks["manifest.json"],
        "readiness_sha256": checks["readiness.json"],
        "config_sha256": config_sha256,
    }
    for role, maximum in [("train", 128), ("validation", 64)]:
        payload = json.loads((directory / f"{role}.json").read_text())
        groups = {}
        for row in payload["rows"]:
            groups.setdefault(row["component_id"], []).append(row)
        chosen = []
        count = 0
        for group in sorted(
            groups,
            key=lambda g: hashlib.sha256(f"2026:{role}:{g}".encode()).hexdigest(),
        ):
            rows = groups[group]
            patients = len({r["patient_id"] for r in rows})
            if count + patients <= maximum:
                chosen.extend(rows)
                count += patients
            if count == maximum:
                break
        if not chosen:
            raise DataValidationError(
                "No whole connected group fits bounded pilot subset"
            )
        spec[role + "_rows"] = chosen
        spec[role + "_parent_manifest_sha256"] = checks[f"{role}.json"]
    return spec


def load_v4_config(path):
    config = tomllib.loads(Path(path).read_text())
    if (
        config.get("schema_version") != 4
        or config.get("protocol") != PROTOCOL
        or config.get("seed") != 2026
        or config.get("class_order") != CLASSES
        or config.get("backbones") != BACKBONES
    ):
        raise DataValidationError(
            "Require fixed v4 protocol, seed and original class/backbone order"
        )
    s = config["split"]
    cal = config["calibration"]
    p = config["pilot"]
    d = config["duplicate_review"]
    if (
        s["fractions"] != [0.65, 0.15, 0.1, 0.1]
        or s["minimum_positive_patients"] != 5
        or s["minimum_negative_patients"] != 5
    ):
        raise DataValidationError("Prespecified split method/floors changed")
    if (
        cal["method"] != "shared_patient_temperature"
        or cal["class_specific_fitting"] is not False
        or cal["policy_fitting"] is not False
        or cal["minimum_independent_groups"] != 50
    ):
        raise DataValidationError("Unsupported calibration/risk-policy plan")
    if (
        p["maximum_epochs"] != 2
        or p["maximum_training_patients"] != 128
        or p["maximum_validation_patients"] != 64
        or p["uses_final_evaluation"] is not False
        or p["backbone"] != "efficientnet_b0"
    ):
        raise DataValidationError("Pilot cap or final-access contract changed")
    if (
        d["policy"] != POLICY
        or d["fixed_scheduled_pairs"] != 1536
        or d["additional_cpu_budget_seconds"] != 7200
        or d["weak_dhash_alone_creates_link"] is not False
    ):
        raise DataValidationError("Review budget, schedule or evidence scope changed")
    selection = config["selection"]
    if (
        selection["role"] != "validation"
        or selection["threshold"] != 0.5
        or selection["final_evaluation_for_selection"] is not False
        or cal["role"] != "calibration"
        or p["full_five_model_training_authorized"] is not False
    ):
        raise DataValidationError(
            "Selection/calibration role or five-model authorization changed"
        )
    return config
