"""Read pinned local receipts only after checking source, nonce and every output digest."""

import json
from pathlib import Path
from trustretina.data.validation import file_hash


def read_stage(bundle):
    bundle = Path(bundle)
    v = json.loads((bundle / "verification.json").read_text())
    s = json.loads((bundle / "submission.json").read_text())
    if v.get("integrity_verified") is not True or any(
        v.get(k) != s.get(k)
        for k in (
            "nonce",
            "source_commit",
            "source_sha256",
            "kernel_version",
            "training",
            "gpu_requested",
        )
    ):
        raise ValueError("Pinned receipt identity mismatch")
    m = json.loads((bundle / "remote-source/kernel-metadata.json").read_text())
    if (
        m["enable_gpu"] != s["gpu_requested"]
        or m["is_private"] is not True
        or file_hash(bundle / "remote-source" / m["code_file"]) != s["source_sha256"]
    ):
        raise ValueError("Remote source/resources mismatch")
    p = bundle / "download" / v.get("report_relative_path", "independent-report.json")
    if (
        file_hash(p) != v["artifact_sha256"]
        or file_hash(p) != p.with_suffix(".sha256").read_text().strip()
    ):
        raise ValueError("Stage report checksum mismatch")
    r = json.loads(p.read_text())
    if (
        r["nonce"] != s["nonce"]
        or r["source_commit"] != s["source_commit"]
        or r["training"] != s["training"]
        or r["gpu_requested"] != s["gpu_requested"]
    ):
        raise ValueError("Stage report/receipt mismatch")
    for name, digest in r.get("output_hashes", {}).items():
        f = (p.parent / name).resolve()
        if (
            not f.is_relative_to(p.parent.resolve())
            or f.is_symlink()
            or file_hash(f) != digest
        ):
            raise ValueError("Stage output changed")
    return r, p.parent, v
