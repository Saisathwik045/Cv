"""Single, release-gated two-epoch GPU execution pilot; no final data or calibration fitting."""

import os, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
from trustretina.data.validation import canonical_hash, file_hash


def validate_pilot_spec(spec):
    if (
        spec.get("scope") != "one_efficientnet_b0_execution_pilot_v4"
        or spec.get("data_checks_passed") is not True
        or spec.get("final_evaluation_access") is not False
    ):
        raise PermissionError(
            "Pilot requires passed data audit; final access is forbidden"
        )
    if (
        spec.get("epochs") != 2
        or spec.get("backbone") != "efficientnet_b0"
        or spec.get("initialization") != "random_unfitted_execution_only"
        or spec.get("batch_size") != 16
    ):
        raise PermissionError("Pilot exceeds fixed execution contract")
    if any(k in spec for k in ("final_rows", "calibration_rows", "all_patients")):
        raise PermissionError("Pilot payload contains forbidden partitions")
    seen = {}
    groups = {}
    paths = {}
    for role, maximum in [("train", 128), ("validation", 64)]:
        rows = spec[role + "_rows"]
        ids = {str(r["patient_id"]) for r in rows}
        if not 1 <= len(ids) <= maximum or len(rows) != 2 * len(ids):
            raise PermissionError(
                "Subset must contain both eyes of bounded patient groups"
            )
        eyes = {}
        for r in rows:
            pid = str(r["patient_id"])
            g = r["component_id"]
            eye = r["eye"]
            path = r["path"]
            labels = tuple(r["labels"])
            if (
                pid in seen
                and seen[pid] != role
                or g in groups
                and groups[g] != role
                or path in paths
            ):
                raise PermissionError(
                    "Patient/group/image crosses pilot roles or repeats"
                )
            if (
                eye not in ("left", "right")
                or len(labels) != 8
                or any(x not in (0, 1) for x in labels)
            ):
                raise PermissionError("Invalid patient/eye/label row")
            eyes.setdefault(pid, {})[eye] = labels
            seen[pid] = role
            groups[g] = role
            paths[path] = role
        if any(
            set(e) != {"left", "right"} or e["left"] != e["right"]
            for e in eyes.values()
        ):
            raise PermissionError("Both original patient-derived eye labels required")
    if (
        len(spec.get("split_sha256", "")) != 64
        or len(spec.get("readiness_sha256", "")) != 64
    ):
        raise PermissionError("Trusted release identities required")
    return paths


def tree_equal(a, b):
    import torch

    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(
            a.detach().cpu(), b.detach().cpu()
        )
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(tree_equal(v, b[k]) for k, v in a.items())
        )
    if isinstance(a, (tuple, list)):
        return (
            type(a) == type(b)
            and len(a) == len(b)
            and all(tree_equal(v, w) for v, w in zip(a, b))
        )
    return a == b


def run_pilot(spec, *, input_root, output, nonce, source_commit):
    allowed = validate_pilot_spec(
        spec
    )  # before CUDA/model initialization or any data access
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from torch.utils.data import DataLoader
    from trustretina.models import create_backbone
    from trustretina.preprocessing import PreprocessConfig
    from trustretina.data.torch_dataset import PatientEyeDataset, collate_eyes
    from trustretina.training.protocol import ProtocolApproval
    from trustretina.training.engine import train_epoch
    from trustretina.training.checkpoint import (
        save_checkpoint,
        load_checkpoint,
        capture_rng,
    )
    from trustretina.evaluation.metrics import aggregate_patient_mean

    start = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "nonce": nonce,
        "source_commit": source_commit,
        "training": True,
        "gpu_requested": True,
        "scope": spec["scope"],
        "pilot_success": False,
        "final_evaluation_access": False,
        "calibration_fitted": False,
        "result_kind": "execution_evidence_only",
        "output_hashes": {},
        "split_sha256": spec["split_sha256"],
        "readiness_sha256": spec["readiness_sha256"],
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Authorized pilot has no CUDA device; no fallback training"
            )
        random.seed(2026)
        np.random.seed(2026)
        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        # Locate the source using one permitted training filename. Never parse data.xlsx,
        # enumerate final labels, or mount any final-manifest kernel outputs.
        relative = Path(spec["train_rows"][0]["path"])
        matches = []
        for path in Path(input_root).rglob(relative.name):
            candidate = path
            for _ in relative.parts:
                candidate = candidate.parent
            if candidate / relative == path:
                matches.append(candidate)
        if len(matches) != 1:
            raise ValueError("Original image root ambiguous")
        root = matches[0]
        preprocessing = PreprocessConfig()
        digests = {}
        datasets = {}
        access = {role: 0 for role in ("train", "validation")}
        for role in ("train", "validation"):
            payload = {
                "schema": "phase2_eye_rows_v1",
                "role": role,
                "rows": spec[role + "_rows"],
            }
            p = output / f"{role}-subset.json"
            p.write_text(json.dumps(payload, sort_keys=True))
            digests[role] = file_hash(p)
            data = PatientEyeDataset(
                p,
                expected_sha256=digests[role],
                root=root,
                role=role,
                preprocessing=preprocessing,
            )
            datasets[role] = data
        generator = torch.Generator().manual_seed(2026)
        val_generator = torch.Generator().manual_seed(2027)
        loaders = {
            "train": DataLoader(
                datasets["train"],
                batch_size=16,
                shuffle=True,
                num_workers=0,
                generator=generator,
                collate_fn=collate_eyes,
            ),
            "validation": DataLoader(
                datasets["validation"],
                batch_size=16,
                shuffle=False,
                num_workers=0,
                generator=val_generator,
                collate_fn=collate_eyes,
            ),
        }
        approval = ProtocolApproval(
            True,
            "bounded_execution_pilot_v4",
            spec["split_sha256"],
            False,
            None,
            tuple(sorted(digests.items())),
        )
        provenance = {
            "backbone": "efficientnet_b0",
            "split_sha256": spec["split_sha256"],
            "config_sha256": spec["config_sha256"],
            "preprocessing_sha256": preprocessing.fingerprint(),
            "seed": 2026,
            "data_kind": "original_odir",
            "pilot_spec_sha256": canonical_hash(spec),
        }

        def predictions(model):
            flags = {m: m.training for m in model.modules()}
            model.eval()
            ys = []
            ps = []
            pids = []
            eyes = []
            comps = []
            try:
                with torch.inference_mode():
                    for batch in loaders["validation"]:
                        if batch["role"] != "validation":
                            raise PermissionError(
                                "Only validation is permitted for execution checks"
                            )
                        out = model(batch["images"].cuda())
                        prob = torch.sigmoid(out)
                        if (
                            out.shape != batch["targets"].shape
                            or out.shape[1] != 8
                            or not torch.isfinite(out).all()
                        ):
                            raise ValueError("Invalid validation output")
                        access["validation"] += len(out)
                        ps.append(prob.cpu().numpy())
                        ys.append(batch["targets"].numpy())
                        pids += batch["patient_ids"]
                        eyes += batch["eyes"]
                        comps += batch["component_ids"]
            finally:
                for m, flag in flags.items():
                    m.training = flag
            return np.concatenate(ys), np.concatenate(ps), pids, eyes, comps

        print(
            json.dumps(
                {
                    "stage": "pilot_start",
                    "training_images": len(datasets["train"]),
                    "validation_images": len(datasets["validation"]),
                }
            ),
            flush=True,
        )
        model = create_backbone("efficientnet_b0").cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        step = 0
        history = []
        reloads = []
        for epoch in range(2):
            result = train_epoch(
                model,
                loaders["train"],
                optimizer,
                approval=approval,
                allow_fitting=True,
                device="cuda",
            )
            step += result["steps"]
            access["train"] += len(datasets["train"])
            if not math.isfinite(result["loss"]) or any(
                not torch.isfinite(p).all() for p in model.parameters()
            ):
                raise ValueError("Nonfinite training loss or weights")
            y, p, pids, eyes, comps = predictions(model)
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": result["loss"],
                    "global_step": step,
                    "validation_execution_check_only": True,
                }
            )
            ck = output / f"epoch-{epoch}.pt"
            digest = save_checkpoint(
                ck,
                model,
                optimizer,
                completed_epoch=epoch,
                global_step=step,
                provenance=provenance,
                generator=generator,
            )
            expected_model = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            expected_optimizer = optimizer.state_dict()
            expected_rng = capture_rng(generator)
            resumed = create_backbone("efficientnet_b0").cuda()
            resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
            state = load_checkpoint(
                ck,
                resumed,
                resumed_optimizer,
                expected_sha256=digest,
                expected_provenance=provenance,
                generator=generator,
            )
            if (
                state["next_epoch"] != epoch + 1
                or state["global_step"] != step
                or not tree_equal(expected_model, resumed.state_dict())
                or not tree_equal(expected_optimizer, resumed_optimizer.state_dict())
                or not tree_equal(expected_rng, capture_rng(generator))
            ):
                raise ValueError("Strict checkpoint/state/RNG reload mismatch")
            _, after, _, _, _ = predictions(resumed)
            if not np.array_equal(p, after) or not tree_equal(
                expected_rng, capture_rng(generator)
            ):
                raise ValueError("Reload predictions/RNG disagree")
            print(
                json.dumps(
                    {
                        "stage": "pilot_epoch_verified",
                        "epoch": epoch,
                        "global_step": step,
                        "finite_loss": True,
                        "strict_reload": True,
                    }
                ),
                flush=True,
            )
            reloads.append(
                {
                    "epoch": epoch,
                    "sha256": digest,
                    "strict_state_reload": True,
                    "rng_restored": True,
                    "next_epoch": state["next_epoch"],
                    "global_step": step,
                    "validation_predictions_exact": True,
                }
            )
            model = resumed
            optimizer = resumed_optimizer
            if epoch == 0:
                report["epoch_boundary_resume_verified"] = True
        maximum_steps = 2 * math.ceil(len(datasets["train"]) / 16)
        if step != maximum_steps:
            raise ValueError("Pilot optimizer-step budget changed")
        patient = aggregate_patient_mean(pids, eyes, y, p, comps)
        np.savez_compressed(
            output / "validation-predictions.npz",
            image_probabilities=p,
            image_targets=y,
            patient_probabilities=patient["probabilities"],
            patient_targets=patient["labels"],
            patient_ids=np.asarray(patient["patient_ids"], dtype=str),
        )
        (output / "training-log.json").write_text(json.dumps(history, indent=2))
        (output / "reload-evidence.json").write_text(json.dumps(reloads, indent=2))
        report.update(
            pilot_success=True,
            epochs_completed=2,
            optimizer_steps=step,
            finite_loss=True,
            strict_reload_verified=True,
            validation_predictions_retrieved=True,
            device=torch.cuda.get_device_name(0),
            training_patients=len(spec["train_rows"]) // 2,
            validation_patients=len(spec["validation_rows"]) // 2,
            accessed_roles=access,
            final_images_opened=0,
            final_labels_opened=0,
        )
    except Exception as e:
        report.update(error_type=type(e).__name__, error=str(e))
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - start)
        report["packages"] = {"torch": str(torch.__version__), "numpy": np.__version__}
        report["output_hashes"] = {
            p.name: file_hash(p) for p in output.iterdir() if p.is_file()
        }
        raw = json.dumps(report, sort_keys=True, indent=2).encode()
        (output / "pilot-report.json").write_bytes(raw)
        (output / "pilot-report.sha256").write_text(hashlib.sha256(raw).hexdigest())
    return report
