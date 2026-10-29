"""Finalize only the fixed recovered schedule; contain uncertainty without inventing duplicates."""

import argparse, json
from pathlib import Path
from trustretina.data.validation import file_hash, canonical_hash
from trustretina.data.contained_release import (
    load_v4_config,
    merge_scheduled,
    group_constraints,
    audit_and_assign,
    publish_release,
    pilot_spec_from_release,
    POLICY,
)
from trustretina.runner.verified_artifacts import read_stage


def finalize(args):
    config = load_v4_config(args.config)
    inventory_path = Path(args.inventory)
    if (
        file_hash(inventory_path)
        != "033d360d46b671fb8690c958ff5285a30e64a4452b5e2327794b5118c954ac41"
    ):
        raise ValueError("Original immutable inventory identity changed")
    inventory = json.loads(inventory_path.read_text())
    prior, _, prior_receipt = read_stage(args.prior_bundle)
    cache, root, cache_receipt = read_stage(args.cache_bundle)
    if (
        cache["status"] != "complete"
        or cache.get("completed_prefix_verified") is not True
        or cache["recovered_completed"] != 422
        or cache["remaining"] != 1114
    ):
        raise ValueError("Fixed selection reconstruction/recovery incomplete")
    selection = json.loads((root / "selection.json").read_text())
    catalog = json.loads((root / "catalog.json").read_text())
    selection_sha = file_hash(root / "selection.json")
    if (
        cache["validation_reused_from_report_sha256"]
        != prior_receipt["artifact_sha256"]
        or prior_receipt["artifact_sha256"] != selection["prior_report_sha256"]
        or prior["images_verified"] != 7000
        or prior["raw_and_pixel_hashes_verified"] is not True
        or prior["controls_passed"] is not True
    ):
        raise ValueError("Completed input validation/control provenance not reusable")
    parts = [
        [
            json.loads(l)
            for l in (root / "recovered-results.jsonl").read_text().splitlines()
            if l
        ]
    ]
    reports = [cache]
    receipt_digests = [cache_receipt["artifact_sha256"]]
    for folder in args.batch_bundle:
        r, b, v = read_stage(folder)
        if (
            r["selection_sha256"] != selection_sha
            or r["policy"] != selection["policy"]
            or r.get("packages") != cache["packages"]
        ):
            raise ValueError("Batch selection/method/environment mismatch")
        parts.append(
            [json.loads(l) for l in (b / "results.jsonl").read_text().splitlines() if l]
        )
        reports.append(r)
        receipt_digests.append(v["artifact_sha256"])
    rows, missing = merge_scheduled(selection, catalog, parts)
    named = json.loads(Path(args.named_cases).read_text())["material_cases"]
    technical = json.loads(Path(args.technical_review).read_text())
    expected = {c["review_index"] for c in named}
    reviewed = {c["review_index"] for c in technical["cases"]}
    review_complete = (
        technical.get("clinical_adjudication") is False
        and expected == reviewed
        and all(
            c["decision"] == "unresolved_contained" and c.get("evidence")
            for c in technical["cases"]
        )
    )
    edges, decisions = group_constraints(inventory, catalog, rows, named_cases=named)
    ledger = json.loads(Path(args.budget).read_text())
    reserved = sum(r["execution_timeout_seconds"] for r in ledger["reservations"])
    actual = sum(r["elapsed_seconds"] for r in reports)
    if actual > reserved:
        raise ValueError("CPU execution exceeded reserved source timeouts")
    manifest, audit = audit_and_assign(
        inventory,
        edges,
        rows,
        missing,
        config,
        validation_reused=True,
        budget_seconds=reserved,
        receipts_verified=True,
        named_review_complete=review_complete,
    )
    audit.update(
        additional_cpu_actual_seconds=actual,
        source_receipt_checksums=receipt_digests,
        selection_sha256=selection_sha,
        recovered_results_recomputed=False,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for name, obj in [
        (
            "policy-v4.json",
            {
                "policy": POLICY,
                "scope": "Fixed 1536-pair bounded review; no claim that every possible duplicate is excluded",
                "edges": edges,
                "decisions": decisions,
                "weak_candidates_default": "sensitivity_only",
                "clinical_adjudication": False,
                "images_deleted": 0,
                "labels_changed": 0,
            },
        ),
        ("readiness.json", audit),
        ("candidate-manifest.json", manifest),
    ]:
        p = output / name
        p.write_text(json.dumps(obj, sort_keys=True, indent=2))
        p.with_suffix(".sha256").write_text(file_hash(p))
    with (output / "scheduled-results.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    if audit["checks_passed"]:
        released = publish_release(
            output / "frozen-splits",
            manifest,
            audit,
            inventory,
            provenance={
                "prior_validation_report_sha256": prior_receipt["artifact_sha256"],
                "selection_sha256": selection_sha,
                "stage_report_checksums": receipt_digests,
                "config_sha256": file_hash(args.config),
                "containment_policy_sha256": file_hash(output / "policy-v4.json"),
                "technical_review_sha256": file_hash(args.technical_review),
            },
        )
        spec = pilot_spec_from_release(output / "frozen-splits", file_hash(args.config))
        (output / "pilot-spec.json").write_text(json.dumps(spec, sort_keys=True))
        (output / "pilot-spec.sha256").write_text(file_hash(output / "pilot-spec.json"))
    print(
        json.dumps(
            {
                k: v
                for k, v in audit.items()
                if k not in ("support", "scheduled_missing")
            },
            indent=2,
        )
    )
    return 0 if audit["checks_passed"] else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "inventory",
        "config",
        "prior-bundle",
        "cache-bundle",
        "named-cases",
        "technical-review",
        "budget",
        "output",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--batch-bundle", action="append", default=[])
    a = p.parse_args()
    try:
        return finalize(a)
    except (ValueError, OSError, KeyError, TypeError) as e:
        print(json.dumps({"blocked": True, "reason": str(e)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
