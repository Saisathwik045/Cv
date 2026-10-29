"""Render named unresolved cases from immutable originals; never adjudicates or fits."""

import argparse, json
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from trustretina.data.validation import file_hash


def render(cases, root, output, *, cases_sha256):
    cases = Path(cases)
    root = Path(root).resolve()
    output = Path(output)
    if file_hash(cases) != cases_sha256:
        raise ValueError("Private case manifest checksum mismatch")
    if not root.is_dir():
        raise ValueError(
            "Original dataset root is required; use the existing CPU execution workflow"
        )
    source = json.loads(cases.read_text())
    rows = source["material_cases"]
    output.mkdir(parents=True, exist_ok=False)
    pages = []
    for i, row in enumerate(rows):
        page = Image.new("RGB", (1100, 640), "white")
        draw = ImageDraw.Draw(page)
        for column, key in enumerate(("a", "b")):
            r = row[key]
            path = (root / r["path"]).resolve()
            if not path.is_relative_to(root) or file_hash(path) != r["sha256"]:
                raise ValueError("Original case image identity mismatch")
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                rgb.thumbnail((530, 530))
                page.paste(rgb, (column * 550 + 10, 35))
            if file_hash(path) != r["sha256"]:
                raise ValueError("Original image changed during review rendering")
            draw.text(
                (column * 550 + 10, 10), f"{r['patient_id']} {r['eye']}", fill="black"
            )
        draw.text(
            (10, 575),
            f"Review index {row['review_index']}; UNRESOLVED. Technical comparison only; no clinical adjudication.",
            fill="black",
        )
        draw.text(
            (10, 600),
            "Keep originals/labels unchanged. New evidence and a versioned decision are required to resolve this case.",
            fill="black",
        )
        name = f"case-{i:03d}.jpg"
        page.save(output / name, quality=95)
        pages.append(
            {
                "case": i,
                "review_index": row["review_index"],
                "image": name,
                "sha256": file_hash(output / name),
                "status": "unresolved",
            }
        )
    index = {
        "schema_version": 3,
        "source_cases_sha256": cases_sha256,
        "clinical_adjudication": False,
        "automatic_clearance": False,
        "images_deleted": 0,
        "labels_changed": 0,
        "pages": pages,
    }
    (output / "index.json").write_text(json.dumps(index, indent=2))
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", required=True)
    p.add_argument("--cases-sha256", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    try:
        r = render(a.cases, a.root, a.output, cases_sha256=a.cases_sha256)
        print(
            json.dumps(
                {"rendered_cases": len(r["pages"]), "automatic_clearance": False}
            )
        )
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
