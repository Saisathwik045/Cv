"""Technical pixel features, without model dependencies or clinical validity claims."""
import numpy as np


def quality_proxies(rgb):
    x = np.asarray(rgb, dtype=float)
    if (
        x.ndim != 3
        or min(x.shape[:2]) < 2
        or x.shape[2] != 3
        or not np.isfinite(x).all()
        or x.min() < 0
        or x.max() > 255
    ):
        raise ValueError("Require RGB [0,255]")
    gray = x.mean(2) / 255.0
    foreground = gray > 8 / 255.0
    if foreground.sum() < 4:
        return {
            "foreground_fraction": float(foreground.mean()),
            "contrast": 0.0,
            "gradient_energy": 0.0,
        }
    # Technical proxies only; no diagnostic/clinical quality claim.
    return {
        "foreground_fraction": float(foreground.mean()),
        "contrast": float(gray[foreground].std()),
        "gradient_energy": float(
            np.mean(np.diff(gray, axis=0) ** 2) + np.mean(np.diff(gray, axis=1) ** 2)
        ),
    }


