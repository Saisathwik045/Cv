"""Fixed-schedule CPU recovery and append-only batches. No model fitting or validation replay."""

import csv, hashlib, json, time, os
from pathlib import Path
from collections import OrderedDict
import numpy as np
from PIL import Image, ImageOps
from trustretina.data.bounded_review import (
    POLICY_V3,
    coarse_content,
    score_pairs,
    select_detail_pairs,
    prepare_detail,
    compare_detail,
)
from trustretina.data.photo_verification import public_photo_record
from trustretina.data.validation import file_hash, canonical_hash
from trustretina.data.review_render import render
from .bounded_runtime import checkpoint_content


def finish(output, report):
    output = Path(output)
    report["output_hashes"] = {
        str(p.relative_to(output)): file_hash(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name not in ("stage-report.json", "stage-report.sha256")
    }
    raw = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode()
    (output / "stage-report.json").write_bytes(raw)
    (output / "stage-report.sha256").write_text(hashlib.sha256(raw).hexdigest())
    return report


def progress(stage, **fields):
    print(json.dumps({"stage": stage, **fields}), flush=True)


def reconstruct(
    spec, prior, cases, *, root, output, nonce, source_commit, maximum_seconds=1750
):
    """Reconstruct missing ranking only; reuse the completed version-2 input validation."""
    import cv2, skimage

    cv2.setNumThreads(1)
    start = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = spec["catalog"]
    coarse = []
    report = {
        "nonce": nonce,
        "source_commit": source_commit,
        "training": False,
        "gpu_requested": False,
        "stage": "cache_reconstruction",
        "status": "incomplete",
        "inventory_sha256": spec["inventory_sha256"],
        "validation_reused_from_report_sha256": spec["prior_report_sha256"],
        "image_validation_repeated": False,
        "policy": POLICY_V3,
        "packages": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scikit_image": skimage.__version__,
            "Pillow": Image.__version__,
        },
    }
    try:
        if (
            report["packages"] != prior["packages"]
            or prior["controls_passed"] is not True
        ):
            raise ValueError("Recovery environment/control contract changed")
        (output / "cases.json").write_text(json.dumps(cases, sort_keys=True))
        render(
            output / "cases.json",
            root,
            output / "comparisons",
            cases_sha256=file_hash(output / "cases.json"),
        )
        # Immutable source version and the original validation receipt bind input identity.
        # Full RGB decoding is needed for exact original coarse features; no second
        # whole-dataset raw/pixel hash validation or workbook parsing is performed.
        for i, r in enumerate(records):
            if time.monotonic() - start > maximum_seconds:
                raise TimeoutError("Recovery budget reached")
            path = (Path(root) / r["path"]).resolve()
            if not path.is_relative_to(Path(root).resolve()):
                raise ValueError("Input escapes original dataset")
            with Image.open(path) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
            coarse.append(coarse_content(rgb))
            if (i + 1) % 250 == 0 or i == len(records) - 1:
                offset = (i // 250) * 250
                chunk = coarse[offset : i + 1]
                np.savez_compressed(
                    output / f"coarse-{offset:05d}.npz",
                    intensity=np.stack([p["intensity"] for p in chunk]),
                    gradient=np.stack([p["gradient"] for p in chunk]),
                )
                progress(
                    "coarse_recovery",
                    images=i + 1,
                    total=len(records),
                    elapsed_seconds=round(time.monotonic() - start),
                )
        hashes = [int(r["dhash"], 16) for r in records]
        pairs = []
        for i, h in enumerate(hashes):
            for j, old in enumerate(hashes[:i]):
                if (
                    (h ^ old).bit_count() <= 2
                    or records[i]["sha256"] == records[j]["sha256"]
                    or records[i]["pixel_sha256"] == records[j]["pixel_sha256"]
                ):
                    pairs.append((j, i))
        if canonical_hash(pairs) != spec["candidate_order_sha256"]:
            raise ValueError("Original candidate order changed")
        original_count = len(pairs)
        known = set(pairs)
        matrix = np.stack([p["intensity"] for p in coarse])
        outside = {}
        for first in range(0, len(records), 128):
            similarity = matrix[first : first + 128] @ matrix.T
            for offset, row in enumerate(similarity):
                i = first + offset
                for j in np.argsort(-row, kind="stable")[:9]:
                    key = tuple(sorted((i, int(j))))
                    if (
                        i != j
                        and records[i]["patient_id"] != records[j]["patient_id"]
                        and key not in known
                    ):
                        outside[key] = float(row[j])
        outside = sorted(
            outside,
            key=lambda p: (
                -outside[p],
                records[p[0]]["sha256"],
                records[p[1]]["sha256"],
            ),
        )[:128]
        for pair in [tuple(p) for p in spec["rejected_pairs"]] + outside:
            key = tuple(sorted(pair))
            if key not in known:
                pairs.append(key)
                known.add(key)
        screen = score_pairs(coarse, pairs)
        mandatory = list(spec["mandatory_indices"]) + list(
            range(original_count, len(pairs))
        )
        selected = select_detail_pairs(pairs, screen, mandatory, maximum=1536)
        checkpoint_content(output, records, coarse, pairs, screen, selected)
        schedule = [
            {
                "review_index": i,
                "pair_index": i if i < original_count else None,
                "a_index": pairs[i][0],
                "b_index": pairs[i][1],
                "candidate": i < original_count,
                "coarse_intensity_correlation": screen[i][0],
                "coarse_gradient_correlation": screen[i][1],
            }
            for i in selected
        ]
        completed = prior["results"]
        seen = set()
        for old, new in zip(completed, schedule):
            if any(
                old[k] != new[k]
                for k in (
                    "review_index",
                    "pair_index",
                    "a_index",
                    "b_index",
                    "candidate",
                )
            ):
                raise ValueError("Recovered schedule does not match completed prefix")
            if (
                max(
                    abs(old[k] - new[k])
                    for k in (
                        "coarse_intensity_correlation",
                        "coarse_gradient_correlation",
                    )
                )
                > 1e-6
            ):
                raise ValueError("Recovered retinal scores changed")
            if old["review_index"] in seen:
                raise ValueError("Repeated completed result")
            seen.add(old["review_index"])
        (output / "selection.json").write_text(
            json.dumps(
                {
                    "policy": POLICY_V3,
                    "catalog_sha256": file_hash(output / "catalog.json"),
                    "prior_report_sha256": spec["prior_report_sha256"],
                    "schedule": schedule,
                    "completed_indices": sorted(seen),
                    "original_candidate_count": original_count,
                },
                sort_keys=True,
            )
        )
        with (output / "recovered-results.jsonl").open("w") as f:
            for row in completed:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        report.update(
            status="complete",
            scheduled=len(schedule),
            recovered_completed=len(completed),
            remaining=len(schedule) - len(completed),
            completed_prefix_verified=True,
        )
    except Exception as e:
        report.update(error_type=type(e).__name__, error=str(e))
    report.update(
        elapsed_seconds=round(time.monotonic() - start),
        coarse_images_reconstructed=len(coarse),
    )
    return finish(output, report)


def run_batch(
    catalog,
    selection,
    indices,
    *,
    root,
    output,
    nonce,
    source_commit,
    selection_sha256,
    maximum_seconds=850,
    expected_packages=None,
):
    import cv2, skimage

    packages = {
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "scikit_image": skimage.__version__,
        "Pillow": Image.__version__,
    }
    if expected_packages is not None and packages != expected_packages:
        raise ValueError("Batch environment differs from validated review controls")
    start = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cache = OrderedDict()
    done = []
    schedule = {r["review_index"]: r for r in selection["schedule"]}
    recovered = set(selection["completed_indices"])
    report = {
        "nonce": nonce,
        "source_commit": source_commit,
        "stage": "remaining_review_batch",
        "training": False,
        "gpu_requested": False,
        "status": "incomplete",
        "selection_sha256": selection_sha256,
        "policy": POLICY_V3,
        "requested_indices": indices,
        "packages": packages,
    }

    def detail(i):
        if i not in cache:
            r = catalog[i]
            path = (Path(root) / r["path"]).resolve()
            if (
                not path.is_relative_to(Path(root).resolve())
                or file_hash(path) != r["sha256"]
            ):
                raise ValueError("Selected image identity changed")
            with Image.open(path) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
            p = prepare_detail(rgb, r["sha256"])
            if p["pixel_sha256"] != r["pixel_sha256"] or file_hash(path) != r["sha256"]:
                raise ValueError("Selected pixels changed")
            cache[i] = p
            if len(cache) > 32:
                cache.popitem(last=False)
        cache.move_to_end(i)
        return cache[i]

    try:
        if (
            len(set(indices)) != len(indices)
            or recovered.intersection(indices)
            or any(i not in schedule for i in indices)
        ):
            raise ValueError("Batch repeats completed work or expands fixed schedule")
        with (output / "results.jsonl").open("a") as f:
            for index in indices:
                if time.monotonic() - start > maximum_seconds:
                    raise TimeoutError("Checkpointed batch time cap")
                row = schedule[index]
                a, b = detail(row["a_index"]), detail(row["b_index"])
                e = compare_detail(a, b, controls_passed=True)
                result = {
                    **row,
                    "a_photo": public_photo_record(a),
                    "b_photo": public_photo_record(b),
                    "independent_evidence": e,
                }
                f.write(json.dumps(result, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
                done.append(index)
                if len(done) % 25 == 0:
                    progress(
                        "resumed_detail",
                        completed=len(done),
                        total=len(indices),
                        elapsed_seconds=round(time.monotonic() - start),
                    )
        report["status"] = "complete"
    except Exception as e:
        report.update(error_type=type(e).__name__, error=str(e))
    report.update(
        completed_indices=done, elapsed_seconds=round(time.monotonic() - start)
    )
    return finish(output, report)
