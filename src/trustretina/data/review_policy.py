"""Diagnostic review v2: verified links, explicit unresolved quarantine, no split approval."""

from collections import Counter
from .duplicates import components

CATEGORIES = (
    "confirmed_duplicate",
    "strongly_supported_probable_copy",
    "unresolved_candidate",
    "evidence_supported_false_positive",
)
POLICY = "photo_review_quarantine_v2"


def review_candidates(patients, pairs, decisions=()):
    """decisions are hash-bound independent evidence keyed by original pair index.

    Not a silent replacement for stage1 splitting. Output is a private diagnostic
    manifest; any new approved experimental split needs a separate reviewed release.
    """
    by_index = {}
    for decision in decisions:
        index = decision["pair_index"]
        if (
            not isinstance(index, int)
            or not 0 <= index < len(pairs)
            or index in by_index
        ):
            raise ValueError("Invalid/repeated review index")
        pair = pairs[index]
        if any(
            decision[k]["sha256"] != pair[k]["sha256"]
            or decision[k]["patient_id"] != pair[k]["patient_id"]
            or decision[k]["eye"] != pair[k]["eye"]
            for k in ("a", "b")
        ):
            raise ValueError("Review evidence identity mismatch")
        evidence = decision["independent_evidence"]
        if evidence["classification"] not in CATEGORIES or not evidence.get("reason"):
            raise ValueError("Missing review decision/reason")
        if evidence["classification"] != "unresolved_candidate" and not evidence.get(
            "controls_passed"
        ):
            raise ValueError("Independent evidence controls did not pass")
        by_index[index] = evidence
    adjudication = []
    links = []
    unresolved = []
    conflicts = []
    for i, pair in enumerate(pairs):
        exact = pair["file_equal"] or pair["pixel_equal"]
        evidence = by_index.get(i)
        category = (
            "confirmed_duplicate"
            if exact
            else evidence["classification"] if evidence else "unresolved_candidate"
        )
        if not exact and category == "confirmed_duplicate":
            raise ValueError("Exact duplicate requires raw/pixel identity")
        linked = category in CATEGORIES[:2]
        if linked:
            links.append(pair)
        if category == "unresolved_candidate":
            unresolved.append(pair)
        if linked and pair["conflicting_patient_labels"]:
            conflicts.append(i)
        adjudication.append(
            {
                "pair_index": i,
                "classification": category,
                "link": linked,
                "decision": (
                    "must_link"
                    if linked
                    else (
                        "quarantine_endpoints"
                        if category == "unresolved_candidate"
                        else "no_link_under_declared_copy_scope"
                    )
                ),
                "reason": (
                    "Exact raw/canonical pixel identity"
                    if exact
                    else (
                        evidence["reason"]
                        if evidence
                        else "No independent verification; weak dHash candidate retained as unresolved"
                    )
                ),
                "independent_evidence": evidence,
            }
        )
    groups = components(patients, links)
    quarantine = {pair[k]["patient_id"] for pair in unresolved for k in ("a", "b")}
    # Even a verified-copy group touching quarantine must stay intact.
    quarantine = {
        pid for group in groups if quarantine.intersection(group) for pid in group
    }
    eligible = sorted({p["patient_id"] for p in patients} - quarantine)
    return {
        "schema_version": 2,
        "policy": POLICY,
        "training_ready": False,
        "clinical_adjudication": False,
        "images_deleted": 0,
        "labels_changed": 0,
        "pairs": adjudication,
        "counts": dict(Counter(p["classification"] for p in adjudication)),
        "must_link_patient_groups": groups,
        "quarantine_patients": sorted(quarantine),
        "eligible_patients": eligible,
        "linked_annotation_conflicts": conflicts,
        "limitations": "Unreviewed and failed independent verification remain unresolved; not evidence of absence of duplicates outside candidate search",
    }


def support_summary(patients, ids, groups):
    chosen = set(ids)
    rows = [p for p in patients if p["patient_id"] in chosen]
    n = len(rows)
    positives = [sum(p["labels"][j] for p in rows) for j in range(8)]
    sizes = sorted(
        (len(set(g) & chosen) for g in groups if set(g) & chosen), reverse=True
    )
    return {
        "patients": n,
        "images": 2 * n,
        "positive": positives,
        "negative": [n - v for v in positives],
        "components": len(sizes),
        "component_size_histogram": dict(sorted(Counter(sizes).items())),
        "largest_component": max(sizes, default=0),
    }


def assert_partition_separation(partitions, groups, quarantine=()):
    assigned = {}
    blocked = set(quarantine)
    for role, ids in partitions.items():
        for pid in ids:
            if pid in assigned or pid in blocked:
                raise ValueError("Patient repeated across partitions or quarantined")
            assigned[pid] = role
    for group in groups:
        roles = {assigned[p] for p in group if p in assigned}
        if len(roles) > 1:
            raise ValueError("Duplicate-connected patient group crosses partitions")
        if roles and any(p not in assigned for p in group):
            raise ValueError("Partially assigned duplicate group")
    return True
