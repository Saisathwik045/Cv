"""Consume verified private CPU evidence; write diagnostics or a new scoped split release.

Never submits a job. A blocked readiness result exits 2 and cannot freeze splits.
"""

import argparse, json
from pathlib import Path
from collections import Counter
from trustretina.data.validation import file_hash, canonical_hash
from trustretina.data.bounded_review import POLICY_V3, material_uncertainty
from trustretina.data.development import (
    load_development_config,
    development_assignment,
    audit_development,
    write_diagnostic,
    freeze_for_pilot,
)


def read_verified(path, digest):
    if file_hash(path) != digest:
        raise ValueError("Private input checksum mismatch")
    return json.loads(Path(path).read_text())


def prepare(args):
    inventory = read_verified(args.inventory, args.inventory_sha256)
    report = read_verified(args.report, args.report_sha256)
    prior = read_verified(args.prior_report, args.prior_report_sha256)
    if inventory.get('data_kind')!='original_odir':raise ValueError('Synthetic/unknown inventory cannot establish real-data readiness')
    receipt = json.loads(Path(args.receipt).read_text())
    if (
        receipt.get("integrity_verified") is not True
        or receipt.get("artifact_sha256") != args.report_sha256
        or receipt.get("nonce") != report["nonce"]
        or receipt.get("source_commit") != report["source_commit"]
    ):
        raise ValueError("Report does not match verified durable job receipt")
    if (
        prior.get("controls_passed") is not True
        or prior.get("raw_hashes_verified") is not True
        or prior.get("independent_dhash_mismatches")
    ):
        raise ValueError("Prior photographic evidence controls or integrity failed")
    if (
        report["inventory_sha256"] != args.inventory_sha256
        or prior["graph_inventory_sha256"] != args.inventory_sha256
        or report["training"] is not False
        or report["calibration_fitted"] is not False
        or report["gpu_requested"] is not False
        or report["policy"] != POLICY_V3
    ):
        raise ValueError("Unexpected evidence source, scope or method policy")
    for name, expected in report["output_hashes"].items():
        target = (Path(args.report).parent / name).resolve()
        if (
            not target.is_relative_to(Path(args.report).parent.resolve())
            or target.is_symlink()
            or file_hash(target) != expected
        ):
            raise ValueError("Independent artifact checksum mismatch")
    patients = inventory["patients"]
    by_id = {p["patient_id"]: p for p in patients}
    original = inventory["duplicate_adjudication"]["pairs"]
    catalog = [
        dict(
            patient_id=p["patient_id"],
            eye=eye,
            **{
                k: r[k]
                for k in ("path", "sha256", "pixel_sha256", "dhash", "width", "height")
            },
        )
        for p in patients
        for eye, r in sorted(p["eyes"].items())
    ]
    prior_evidence = {
        r["pair_index"]: r["independent_evidence"]
        for r in prior["results"]
        if r["candidate"]
    }
    newer = {}
    outside = []
    review_conflicts = []
    for row in report["results"]:
        a, b = catalog[row["a_index"]], catalog[row["b_index"]]
        e = row["independent_evidence"]
        index = row["pair_index"]
        if (
            row["a_photo"]["raw_sha256"] != a["sha256"]
            or row["b_photo"]["raw_sha256"] != b["sha256"]
        ):
            raise ValueError("Detailed comparison endpoint identity mismatch")
        if index is None:
            outside.append((row, a, b))
            continue
        if any(
            original[index][key]["sha256"] != r["sha256"]
            for key, r in [("a", a), ("b", b)]
        ):
            raise ValueError("Original pair index binding mismatch")
        if prior_evidence.get(index, {}).get(
            "classification"
        ) == "evidence_supported_false_positive" and e.get("link"):
            review_conflicts.append(index)
        newer[index] = e
    links = []
    adjudication = []
    for index, pair in enumerate(original):
        old = prior_evidence.get(index)
        new = newer.get(index)
        if pair["file_equal"] or pair["pixel_equal"]:
            category = "confirmed_duplicate"
            evidence = None
            reason = "Verified original byte/pixel identity"
        elif old and old.get("link"):
            category = old["classification"]
            evidence = old
            reason = "Preserve independently supported prior copy evidence; later verification failure cannot clear a link"
        elif new and new.get("link"):
            category = new["classification"]
            evidence = new
            reason = new["reason"]
        elif old and old["classification"] == "evidence_supported_false_positive":
            category = old["classification"]
            evidence = old
            reason = old["reason"]
        else:
            category = "unresolved_candidate"
            evidence = new or old
            reason = (
                evidence["reason"]
                if evidence
                else "Weak dHash candidate retained for sensitivity; no independent detailed adjudication"
            )
        linked = category in ("confirmed_duplicate", "strongly_supported_probable_copy")
        if linked:
            links.append(pair)
        adjudication.append(
            {
                "pair_index": index,
                "classification": category,
                "decision": (
                    "must_link"
                    if linked
                    else (
                        "supported_no_link"
                        if category == "evidence_supported_false_positive"
                        else "unresolved_sensitivity"
                    )
                ),
                "reason": reason,
                "independent_evidence": evidence,
            }
        )
    for row, a, b in outside:
        if row["independent_evidence"].get("link"):
            links.append(
                {
                    "a": a,
                    "b": b,
                    "file_equal": a["sha256"] == b["sha256"],
                    "pixel_equal": a["pixel_sha256"] == b["pixel_sha256"],
                    "conflicting_patient_labels": by_id[a["patient_id"]]["labels"]
                    != by_id[b["patient_id"]]["labels"],
                }
            )
    if report.get("material_cases_file"):
        material = json.loads(
            (Path(args.report).parent / report["material_cases_file"]).read_text()
        )
    else:
        # A wall-time limited result is still useful evidence, but incomplete
        # coverage cannot be converted into approval. Identify its reviewed cases.
        material = []
        for row in report["results"]:
            a, b = catalog[row["a_index"]], catalog[row["b_index"]]
            if material_uncertainty(
                (a["patient_id"], b["patient_id"]),
                (
                    row["coarse_intensity_correlation"],
                    row["coarse_gradient_correlation"],
                ),
                row["independent_evidence"],
            ):
                material.append(
                    {
                        "a": a,
                        "b": b,
                        "pair_index": row["pair_index"],
                        "review_index": row["review_index"],
                        "detail_evidence": row["independent_evidence"],
                        "reason": "Material reviewed photographic uncertainty remains; bounded pass incomplete",
                    }
                )
    linked_pairs = {tuple(sorted((p["a"]["sha256"], p["b"]["sha256"]))) for p in links}
    supported_false_positive_indices = {
        i
        for i, e in prior_evidence.items()
        if e["classification"] == "evidence_supported_false_positive"
        and i not in review_conflicts
    }
    material = [
        r
        for r in material
        if r.get("pair_index") not in supported_false_positive_indices
        and tuple(sorted((r["a"]["sha256"], r["b"]["sha256"]))) not in linked_pairs
    ]
    config = load_development_config(args.config)
    manifest = development_assignment(inventory, links, config)
    if canonical_hash(manifest) != canonical_hash(
        development_assignment(inventory, links, config)
    ):
        raise ValueError("Deterministic replay mismatch")
    audit = audit_development(
        manifest,
        inventory,
        links,
        material,
        review_complete=report.get("status") == "bounded_pass_complete",
        controls_passed=report.get("controls_passed") is True,
        raw_verified=report.get("raw_and_pixel_hashes_verified") is True
        and report.get("images_verified") == 2 * len(patients),
    )
    if review_conflicts:
        audit["checks_passed"] = audit["pilot_ready"] = False
        audit["blockers"].append("independent_review_decisions_conflict")
    audit.update(
        deterministic_replay_verified=True,
        inventory_sha256=args.inventory_sha256,
        report_sha256=args.report_sha256,
        prior_report_sha256=args.prior_report_sha256,
        config_sha256=file_hash(args.config),
        new_detailed_classifications=dict(
            Counter(
                r["independent_evidence"]["classification"] for r in report["results"]
            )
        ),
        candidate_classifications=dict(
            Counter(r["classification"] for r in adjudication)
        ),
        verified_links=len(links),
        review_conflicts=review_conflicts,
    )
    out = write_diagnostic(args.output, manifest, audit)
    evidence = {
        "schema_version": 3,
        "policy": "verified_links_weak_candidates_sensitivity_v3",
        "inventory_sha256": args.inventory_sha256,
        "report_sha256": args.report_sha256,
        "prior_report_sha256": args.prior_report_sha256,
        "pairs": adjudication,
        "outside_hash_verified_links": [
            p for p in links if "patient_labels" not in p["a"]
        ],
        "material_cases": material,
        "patient_groups": manifest["patient_groups"],
        "images_deleted": 0,
        "labels_changed": 0,
        "clinical_adjudication": False,
        "default_quarantine": False,
        "all_weak_edges_must_link": False,
    }
    path = out / "adjudication-v3.json"
    path.write_text(json.dumps(evidence, sort_keys=True))
    path.with_suffix(".sha256").write_text(file_hash(path))
    if audit["checks_passed"]:
        freeze_for_pilot(
            out / "frozen-splits",
            manifest,
            audit,
            inventory,
            config_sha256=file_hash(args.config),
            evidence_sha256=file_hash(path),
        )
    else:
        (out / "pilot-blocked.json").write_text(
            json.dumps(
                {
                    "conditional_user_authorization": True,
                    "submission_attempted": False,
                    "job_id": None,
                    "blockers": audit["blockers"],
                    "readiness_sha256": file_hash(out / "readiness.json"),
                    "full_training_ready": False,
                },
                indent=2,
            )
        )
    summary = {
        k: audit[k]
        for k in (
            "checks_passed",
            "pilot_ready",
            "full_training_ready",
            "blockers",
            "material_cases",
            "material_cross_partition_cases",
            "verified_links",
            "candidate_classifications",
            "support",
        )
    }
    print(json.dumps(summary, indent=2))
    return 0 if audit["checks_passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "inventory",
        "inventory-sha256",
        "report",
        "report-sha256",
        "receipt",
        "prior-report",
        "prior-report-sha256",
        "config",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        return prepare(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": str(exc),
                    "exception": type(exc).__name__,
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
