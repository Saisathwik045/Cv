"""Private CPU review entry point. No model fitting, network calls or split publication."""

import csv, hashlib, json, time
from pathlib import Path
from collections import Counter, OrderedDict
import numpy as np
from PIL import Image, ImageOps
from trustretina.data.validation import file_hash, canonical_hash, parse_workbook
from trustretina.data.bounded_review import (
    POLICY_V3,
    coarse_content,
    score_pairs,
    select_detail_pairs,
    prepare_detail,
    compare_detail,
    material_uncertainty,
)
from trustretina.data.photo_verification import public_photo_record


def run_review(spec, *, nonce, source_commit, input_root, output):
    import cv2, skimage

    cv2.setNumThreads(1)
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = spec["catalog"]
    coarse = []
    cache = OrderedDict()
    results = []
    controls = []
    screen = []
    pairs = []
    material = []
    report = {
        "nonce": nonce,
        "source_commit": source_commit,
        "training": False,
        "calibration_fitted": False,
        "gpu_requested": False,
        "training_ready": False,
        "new_experimental_splits_published": False,
        "inventory_sha256": spec["inventory_sha256"],
        "policy": POLICY_V3,
        "status": "incomplete",
        "output_hashes": {},
        "packages": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scikit_image": skimage.__version__,
            "Pillow": Image.__version__,
        },
    }

    def budget():
        if time.monotonic() - started > POLICY_V3["maximum_seconds"]:
            raise TimeoutError(
                "Bounded review wall-time exhausted; no split/pilot approval"
            )

    def progress(stage, **counts):
        print(
            json.dumps(
                {
                    "stage": stage,
                    **counts,
                    "elapsed_seconds": round(time.monotonic() - started),
                }
            ),
            flush=True,
        )

    def read_rgb(i):
        record = records[i]
        path = (Path(input_root) / record["path"]).resolve()
        if (
            not path.is_relative_to(Path(input_root).resolve())
            or file_hash(path) != record["sha256"]
        ):
            raise ValueError("Raw input identity mismatch")
        with Image.open(path) as source:
            rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        actual = hashlib.sha256(
            str((rgb.shape[1], rgb.shape[0])).encode() + rgb.tobytes()
        ).hexdigest()
        if actual != record["pixel_sha256"] or file_hash(path) != record["sha256"]:
            raise ValueError("Canonical pixel mismatch or raw mutation")
        return rgb

    def detail(i):
        budget()
        if i not in cache:
            p = prepare_detail(read_rgb(i), records[i]["sha256"])
            cache[i] = p
            if len(cache) > 24:
                cache.popitem(last=False)
        cache.move_to_end(i)
        return cache[i]

    try:
        workbook = Path(input_root) / spec["workbook_path"]
        if file_hash(workbook) != spec["workbook_sha256"]:
            raise ValueError("Authoritative workbook checksum mismatch")
        patients, sheet = parse_workbook(workbook)
        if canonical_hash(patients) != spec["parsed_patients_sha256"]:
            raise ValueError("Workbook labels/references changed")
        report.update(
            workbook_verified=True,
            parsed_patients_verified=True,
            patients=len(patients),
            sheet=sheet,
        )
        for i, r in enumerate(records):
            budget()
            rgb = read_rgb(i)
            coarse.append(coarse_content(rgb))
            small = np.asarray(
                Image.fromarray(rgb)
                .convert("L")
                .resize((9, 8), Image.Resampling.LANCZOS)
            )
            dhash = f"{int.from_bytes(np.packbits((small[:,:-1]>small[:,1:]).ravel()).tobytes(),'big'):016x}"
            if dhash != r["dhash"]:
                raise ValueError("Independent dHash implementation mismatch")
            if (i + 1) % 250 == 0:
                progress("retinal_content_scan", images=i + 1, total=len(records))
        report.update(
            images_verified=len(coarse),
            raw_and_pixel_hashes_verified=True,
            independent_dhash_mismatches=0,
        )
        # Same enumeration/order as immutable v1 inventory; dHash only schedules candidates.
        hashes = [int(r["dhash"], 16) for r in records]
        for i, h in enumerate(hashes):
            for j, old in enumerate(hashes[:i]):
                if (
                    (h ^ old).bit_count() <= 2
                    or records[i]["sha256"] == records[j]["sha256"]
                    or records[i]["pixel_sha256"] == records[j]["pixel_sha256"]
                ):
                    pairs.append((j, i))
        if canonical_hash(pairs) != spec["candidate_order_sha256"]:
            raise ValueError("Original candidate enumeration mismatch")
        original_count = len(pairs)
        known = set(pairs)
        # Independent foreground nearest-neighbour retrieval, including pairs rejected
        # by dHash; labels and partition sizes never enter retrieval or ranking.
        matrix = np.stack([r["intensity"] for r in coarse])
        outside = {}
        for first in range(0, len(records), 128):
            budget()
            similarity = matrix[first : first + 128] @ matrix.T
            for offset, row in enumerate(similarity):
                i = first + offset
                nearest = np.argsort(-row, kind="stable")[:9]
                for j in nearest:
                    key = tuple(sorted((i, int(j))))
                    if (
                        i != j
                        and records[i]["patient_id"] != records[j]["patient_id"]
                        and key not in known
                    ):
                        outside[key] = float(row[j])
        outside = sorted(
            outside,
            key=lambda pair: (
                -outside[pair],
                records[pair[0]]["sha256"],
                records[pair[1]]["sha256"],
            ),
        )[:128]
        extras = []
        for pair in [tuple(x) for x in spec["rejected_pairs"]] + outside:
            key = tuple(sorted(pair))
            if key not in known:
                extras.append(key)
                known.add(key)
        pairs.extend(extras)
        screen = score_pairs(coarse, pairs)
        mandatory = list(spec["mandatory_indices"]) + list(
            range(original_count, len(pairs))
        )
        selection = select_detail_pairs(
            pairs, screen, mandatory, maximum=POLICY_V3["maximum_detail_pairs"]
        )
        # Four label-blind informative sources. Synthetic derivative controls are not
        # dataset duplicates and never enter patient groups or split counts.
        control_sources = []
        for i in sorted(range(len(records)), key=lambda i: records[i]["sha256"]):
            a = detail(i)
            if not a["low_information"] and a["features"] >= 40:
                control_sources.append(i)
            if len(control_sources) == 4:
                break
        if len(control_sources) < 4:
            raise ValueError("Insufficient informative copy controls")
        for i in control_sources:
            rgb = read_rgb(i)
            a = detail(i)
            h, w = rgb.shape[:2]
            small = cv2.resize(rgb, (max(2, w // 2), max(2, h // 2)))
            _, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            jpeg = cv2.cvtColor(
                cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
            for name, image in [
                ("resize_jpeg90", jpeg),
                (
                    "mild_exposure",
                    np.clip(rgb.astype(float) * 1.01 + 1, 0, 255).astype(np.uint8),
                ),
                ("black_padding", np.pad(rgb, ((30, 30), (40, 40), (0, 0)))),
            ]:
                evidence = compare_detail(a, prepare_detail(image))
                controls.append(
                    {
                        "source_index": i,
                        "transform": name,
                        "passed": evidence["link"],
                        "evidence": evidence,
                    }
                )
        low = compare_detail(
            prepare_detail(np.full((512, 512, 3), 50, np.uint8)),
            prepare_detail(np.full((512, 512, 3), 51, np.uint8)),
        )
        controls.append(
            {
                "transform": "low_information_negative",
                "passed": not low["link"],
                "evidence": low,
            }
        )
        passed = all(c["passed"] for c in controls)
        report["controls_passed"] = passed
        progress("bounded_detail_start", pairs=len(selection), controls_passed=passed)
        for k, index in enumerate(selection):
            budget()
            i, j = pairs[index]
            a, b = detail(i), detail(j)
            e = compare_detail(a, b, controls_passed=passed)
            row = {
                "pair_index": index if index < original_count else None,
                "review_index": index,
                "a_index": i,
                "b_index": j,
                "candidate": index < original_count,
                "coarse_intensity_correlation": screen[index][0],
                "coarse_gradient_correlation": screen[index][1],
                "a_photo": public_photo_record(a),
                "b_photo": public_photo_record(b),
                "independent_evidence": e,
            }
            results.append(row)
            if (k + 1) % 25 == 0:
                progress("bounded_detail", pairs=k + 1, total=len(selection))
        report.update(
            status="bounded_pass_complete",
            candidate_pairs=original_count,
            outside_dhash_pairs=len(extras),
            detail_budget=len(selection),
            detail_completed=len(results),
            detail_candidates=sum(r["candidate"] for r in results),
        )
        evidence = {r["review_index"]: r["independent_evidence"] for r in results}
        prior_possible = set(spec["prior_possible_copy_indices"])
        screen_path = output / "screening.csv"
        with screen_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "review_index",
                    "original_pair_index",
                    "a_index",
                    "b_index",
                    "dhash_distance",
                    "retinal_intensity_correlation",
                    "retinal_gradient_correlation",
                    "classification",
                    "reason",
                    "material_review_required",
                ]
            )
            for index, ((i, j), score) in enumerate(zip(pairs, screen)):
                e = evidence.get(index)
                category = e["classification"] if e else "unresolved_candidate"
                reason = (
                    e["reason"]
                    if e
                    else "Outside bounded detailed-review budget; screening similarity is not adjudication"
                )
                cross = records[i]["patient_id"] != records[j]["patient_id"]
                pending = cross and material_uncertainty(
                    (records[i]["patient_id"], records[j]["patient_id"]),
                    score,
                    e,
                    prior_possible_copy=index in prior_possible,
                )
                if pending:
                    material.append(
                        {
                            "review_index": index,
                            "pair_index": index if index < original_count else None,
                            "a": records[i],
                            "b": records[j],
                            "coarse_scores": score,
                            "detail_evidence": e,
                            "reason": "Potential photographic copy remains unverified under bounded independent review; not cleared by hash or failed feature matching",
                        }
                    )
                writer.writerow(
                    [
                        index,
                        index if index < original_count else "",
                        i,
                        j,
                        (hashes[i] ^ hashes[j]).bit_count(),
                        *score,
                        category,
                        reason,
                        pending,
                    ]
                )
        report.update(
            material_uncertainty_count=len(material),
            material_cases_file="material-cases.json",
            screening_file="screening.csv",
            classification_counts=dict(
                Counter(r["independent_evidence"]["classification"] for r in results)
            ),
            outcome=(
                "blocked_material_duplicate_uncertainty"
                if material or not passed
                else "needs_independent_protocol_audit"
            ),
        )
        (output / "material-cases.json").write_text(
            json.dumps(material, sort_keys=True)
        )
        (output / "catalog.json").write_text(json.dumps(records, sort_keys=True))
    except Exception as exc:
        report.update(
            error_type=type(exc).__name__,
            error=str(exc),
            outcome="blocked_incomplete_review",
            images_verified=len(coarse),
            detail_completed=len(results),
        )
    finally:
        report.update(
            controls=controls,
            results=results,
            elapsed_seconds=round(time.monotonic() - started),
            clinical_adjudication=False,
            images_deleted=0,
            labels_changed=0,
        )
        for path in sorted(output.iterdir()):
            if path.is_file():
                report["output_hashes"][str(path.relative_to(output))] = file_hash(path)
        raw = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode()
        (output / "independent-report.json").write_bytes(raw)
        (output / "independent-report.sha256").write_text(
            hashlib.sha256(raw).hexdigest()
        )
        progress(
            "bounded_review_complete",
            outcome=report.get("outcome"),
            detail_completed=len(results),
            material_cases=len(material),
        )
    return report
