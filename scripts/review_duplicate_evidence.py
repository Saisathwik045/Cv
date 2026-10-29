import json, hashlib
from pathlib import Path
from trustretina.data.review_policy import review_candidates, support_summary
from trustretina.data.duplicates import components
import argparse

parser = argparse.ArgumentParser(
    description="Produce private diagnostic quarantine/sensitivity manifests; never publish experimental splits"
)
parser.add_argument("--inventory", type=Path, required=True)
parser.add_argument("--inventory-sha256", required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--report-sha256", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
base = args.output
base.mkdir(parents=True, exist_ok=False)
p = args.inventory
if hashlib.sha256(p.read_bytes()).hexdigest() != args.inventory_sha256:
    raise ValueError("Inventory checksum mismatch")
inventory = json.loads(p.read_text())
patients = inventory["patients"]
pairs = inventory["duplicate_adjudication"]["pairs"]
rp = args.report
report = json.loads(rp.read_text())
if hashlib.sha256(rp.read_bytes()).hexdigest() != args.report_sha256:
    raise ValueError("Independent report checksum mismatch")
if report["graph_inventory_sha256"] != args.inventory_sha256:
    raise ValueError("Evidence inventory binding mismatch")
if (
    report["training"] is not False
    or report["calibration_fitted"] is not False
    or report["gpu_requested"] is not False
):
    raise ValueError("Unexpected execution scope")
if (
    not report["controls_passed"]
    or not report["raw_hashes_verified"]
    or report["independent_dhash_mismatches"]
):
    raise ValueError("Independent controls or input integrity failed")
review = review_candidates(
    patients, pairs, [r for r in report["results"] if r["candidate"]]
)
review["provenance"] = {
    "inventory_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    "independent_report_sha256": hashlib.sha256(rp.read_bytes()).hexdigest(),
    "kernel_nonce": report["nonce"],
    "kernel_source_commit": report["source_commit"],
    "dataset_fingerprint": inventory["fingerprint"],
}
# Preserve original pair identity without storing an altered inventory; indexes bind
# to immutable inventory bytes, independent evidence additionally binds raw hashes.
path = base / "adjudication-v2.json"
path.write_text(json.dumps(review, sort_keys=True))
path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
allids = [p["patient_id"] for p in patients]
exact = components(patients, [p for p in pairs if p["file_equal"] or p["pixel_equal"]])
weak = components(patients, pairs)
summary = {
    "training_ready": False,
    "class_order": inventory["class_order"],
    "candidate_counts": review["counts"],
    "linked_annotation_conflicts": len(review["linked_annotation_conflicts"]),
    "all_patients": support_summary(
        patients, allids, review["must_link_patient_groups"]
    ),
    "quarantine": support_summary(
        patients, review["quarantine_patients"], review["must_link_patient_groups"]
    ),
    "eligible": support_summary(
        patients, review["eligible_patients"], review["must_link_patient_groups"]
    ),
    "sensitivity_exact_only": support_summary(patients, allids, exact),
    "sensitivity_all_weak_links": support_summary(patients, allids, weak),
    "adjudication_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "new_splits_published": False,
}
(base / "feasibility-v2.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
