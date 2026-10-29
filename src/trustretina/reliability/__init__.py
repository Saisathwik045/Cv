"""Reliability interfaces; numeric quality/calibration analysis does not require Torch."""

from .quality import quality_proxies

__all__ = ["mc_dropout", "quality_proxies", "ReliabilityPolicy", "fit_reliability"]


def __getattr__(name):
    if name in ("mc_dropout", "ReliabilityPolicy", "fit_reliability"):
        from importlib import import_module

        value = getattr(import_module(".scoring", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
