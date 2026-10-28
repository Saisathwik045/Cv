"""Generate a declarative job inventory; this does not execute jobs."""
import hashlib
import json


def execution_plan(config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    jobs = []
    for backbone in config["experiment"]["backbones"]:
        jobs.append({"backbone": backbone, "run_key": f"{backbone}-{digest[:12]}",
                     "stages": ["train", "validation_predictions", "calibration_predictions",
                                "fit_calibration", "freeze_policy", "final_evaluation", "gradcam"],
                     "status": "not_implemented"})
    return {"schema_version": 1, "kind": "declarative_plan_only", "config_sha256": digest,
            "prerequisites": ["validated_dataset", "approved_patient_manifest", "verified_gpu_runner"],
            "jobs": jobs}
