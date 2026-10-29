"""Hash-verified read-only split adapter; no implicit partition creation or data mutation."""

from pathlib import Path
import json
import torch
from PIL import Image
from trustretina.preprocessing import preprocess, PreprocessConfig
from trustretina.training.protocol import sha256


class PatientEyeDataset(torch.utils.data.Dataset):
    def __init__(
        self, manifest_path, *, expected_sha256, root, role, preprocessing=None
    ):
        if sha256(manifest_path) != expected_sha256:
            raise ValueError("Immutable manifest checksum mismatch")
        manifest = json.loads(Path(manifest_path).read_text())
        if (
            manifest.get("schema") != "phase2_eye_rows_v1"
            or manifest.get("role") != role
            or role
            not in (
                "train",
                "validation",
                "calibration",
                "calibration_fit",
                "calibration_policy",
                "final_evaluation",
            )
        ):
            raise ValueError(
                "Explicit phase2 role/row schema required; historical manifests need reviewed conversion"
            )
        self.rows = manifest["rows"]
        self.root = Path(root).resolve()
        self.role = role
        self.preprocessing = preprocessing or PreprocessConfig()
        self.manifest_sha256 = expected_sha256
        seen = set()
        patients = {}
        groups = {}
        for row in self.rows:
            key = (str(row["patient_id"]), row["eye"])
            labels = tuple(row["labels"])
            if (
                key in seen
                or row["eye"] not in ("left", "right")
                or len(labels) != 8
                or any(x not in (0, 1) for x in labels)
            ):
                raise ValueError("Invalid duplicate row/eye/labels")
            if key[0] in patients and patients[key[0]] != labels:
                raise ValueError("Patient label mismatch")
            if key[0] in groups and groups[key[0]] != row["component_id"]:
                raise ValueError("Patient duplicate-group mismatch")
            seen.add(key)
            patients[key[0]] = labels
            groups[key[0]] = row["component_id"]
        if not seen or any(
            (pid, eye) not in seen for pid in patients for eye in ("left", "right")
        ):
            raise ValueError("Both eyes must be present")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["path"]).resolve()
        if not path.is_relative_to(self.root) or sha256(path) != row["raw_sha256"]:
            raise ValueError("Image escaped root or immutable image checksum changed")
        with Image.open(path) as image:
            tensor, _ = preprocess(image, self.preprocessing, role=self.role)
        return {
            "images": tensor,
            "targets": torch.tensor(row["labels"], dtype=torch.float32),
            "patient_id": str(row["patient_id"]),
            "eye": row["eye"],
            "component_id": row["component_id"],
            "role": self.role,
            "manifest_sha256": self.manifest_sha256,
        }


def collate_eyes(rows):
    if (
        not rows
        or len({r["role"] for r in rows}) != 1
        or len({r["manifest_sha256"] for r in rows}) != 1
    ):
        raise ValueError("Cannot mix partition manifests")
    return {
        "images": torch.stack([r["images"] for r in rows]),
        "targets": torch.stack([r["targets"] for r in rows]),
        "patient_ids": [r["patient_id"] for r in rows],
        "eyes": [r["eye"] for r in rows],
        "component_ids": [r["component_id"] for r in rows],
        "role": rows[0]["role"],
        "manifest_sha256": rows[0]["manifest_sha256"],
    }
