"""Read published aggregate evidence only. No private patient artifacts."""
import csv
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]


def load_evidence(root=ROOT):
    root = Path(root)
    path = root / 'results/demo-v8/evidence-hashes.json'
    pinned = json.loads(path.read_text())
    for name, expected in pinned.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Published aggregate evidence checksum mismatch')
    with (root / 'results/full-v5/comparison.csv').open() as f:
        validation = list(csv.DictReader(f))
    final = json.loads((root / 'results/final-v7/aggregate-summary.json').read_text())
    return validation, final, (root / 'results/final-v7/REPORT.md').read_text()
