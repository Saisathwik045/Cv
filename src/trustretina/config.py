"""Validated, portable experiment configuration (no dataset reads)."""
import math
from pathlib import Path
import tomllib

CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]
BACKBONES = ["efficientnet_b0", "resnet50", "densenet121", "convnext_tiny", "efficientnet_b2"]
PARTITIONS = ["train", "validation", "calibration", "final_evaluation"]


def load_config(path):
    path = Path(path).resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("schema_version") != 1 or config.get("class_order") != CLASSES:
        raise ValueError("Require schema 1 and class order N,D,G,C,A,H,M,O")
    if config['data'].get('kind', 'original_odir') not in ('original_odir', 'synthetic_fixture'):
        raise ValueError('data.kind must be original_odir or synthetic_fixture')
    if config["data"]["label_source"] != "patient":
        raise ValueError("ODIR source labels must be described as patient labels")
    if Path(config["data"]["labels"]).name != "data.xlsx":
        raise ValueError("Original data.xlsx is the authoritative label source")
    if config['data'].get('duplicate_policy', 'reject') not in ('reject', 'conservative_patient_components_v1'):
        raise ValueError('Unknown duplicate handling policy')
    split = config["split"]
    fractions = split["fractions"]
    if (split["names"] != PARTITIONS or len(fractions) != 4
            or any(not math.isfinite(x) or not 0 < x < 1 for x in fractions)
            or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9)):
        raise ValueError("Require four positive patient partitions summing to 1")
    for key in ("minimum_positive_patients", "minimum_negative_patients"):
        if type(split[key]) is not int or split[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["experiment"]["backbones"] != BACKBONES:
        raise ValueError("Require all five backbones in the fixed comparison order")
    if config["experiment"]["loss"] != "BCEWithLogitsLoss":
        raise ValueError("Use BCEWithLogitsLoss for eight independent labels")
    if config["experiment"]["selection_metric"] != "validation_macro_f1":
        raise ValueError("Select models on validation_macro_f1 only")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer")
    experiment = config["experiment"]
    for key in ("image_size", "max_epochs", "batch_size"):
        if type(experiment[key]) is not int or experiment[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < experiment["initial_threshold"] < 1:
        raise ValueError("initial_threshold must lie inside (0,1)")
    if not math.isfinite(experiment["learning_rate"]) or experiment["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    reliability = config["reliability"]
    if type(reliability["mc_dropout_passes"]) is not int or reliability["mc_dropout_passes"] < 2:
        raise ValueError("MC Dropout requires at least two passes")
    if not 0 < reliability["dropout_probability"] < 1:
        raise ValueError("dropout_probability must lie inside (0,1)")
    for key in ("root",):
        config["data"][key] = str((path.parent / config["data"][key]).resolve())
    config["paths"]["artifacts"] = str((path.parent / config["paths"]["artifacts"]).resolve())
    return config
