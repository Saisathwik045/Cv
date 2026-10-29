"""Shared deterministic image geometry; no model/data fitting or raw-file mutation."""

from dataclasses import dataclass, asdict
import hashlib, json
from PIL import Image, ImageOps
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    size: int = 224
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    geometry: str = "exif_rgb_letterbox_v1"
    training_horizontal_flip: bool = False

    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


def preprocess(image, config=PreprocessConfig(), *, role="validation", rng=None):
    import torch

    if (
        config.size < 32
        or config.geometry != "exif_rgb_letterbox_v1"
        or len(config.mean) != 3
        or len(config.std) != 3
        or not np.isfinite(config.mean).all()
        or not np.isfinite(config.std).all()
        or min(config.std) <= 0
    ):
        raise ValueError("Invalid declared shared preprocessing")
    if role not in (
        "train",
        "validation",
        "calibration",
        "calibration_fit",
        "calibration_policy",
        "final_evaluation",
        "inference",
    ):
        raise ValueError("Unknown role")
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    width, height = rgb.size
    scale = min(config.size / width, config.size / height)
    resized = rgb.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    left = (config.size - resized.width) // 2
    top = (config.size - resized.height) // 2
    canvas = Image.new(
        "RGB", (config.size, config.size), tuple(round(v * 255) for v in config.mean)
    )
    canvas.paste(resized, (left, top))
    flip = False
    if role == "train" and config.training_horizontal_flip:
        if rng is None:
            raise ValueError(
                "Training augmentation requires explicit checkpointable RNG"
            )
        flip = bool(rng.random() < 0.5)
        if flip:
            canvas = ImageOps.mirror(canvas)
    array = np.asarray(canvas, dtype=np.float32) / 255
    array = (array - np.asarray(config.mean, np.float32)) / np.asarray(
        config.std, np.float32
    )
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    return tensor, {
        "original_size": [width, height],
        "resized_size": [resized.width, resized.height],
        "padding": [
            left,
            top,
            config.size - left - resized.width,
            config.size - top - resized.height,
        ],
        "horizontal_flip": flip,
        "preprocessing_sha256": config.fingerprint(),
    }
