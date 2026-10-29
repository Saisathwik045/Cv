"""CPU forward/state/contract tests. Never trains or fits calibration parameters."""

import unittest, tempfile, random, json
from pathlib import Path

try:
    import torch
    import torchvision
    import numpy as np
    from PIL import Image
    from trustretina.models import create_backbone
    from trustretina.preprocessing import preprocess, PreprocessConfig
    from trustretina.evaluation.metrics import (
        multilabel_metrics,
        aggregate_patient_mean,
    )
    from trustretina.evaluation.runner import evaluate
    from trustretina.training.engine import (
        train_epoch,
        training_loss,
        select_checkpoint,
        select_backbone,
    )
    from trustretina.training.checkpoint import save_checkpoint, load_checkpoint
    from trustretina.training.protocol import ProtocolNotReady, sha256
    from trustretina.calibration import TemperatureCalibration, fit_temperature
    from trustretina.reliability import mc_dropout, ReliabilityPolicy, fit_reliability
    from trustretina.data.torch_dataset import PatientEyeDataset, collate_eyes
    from trustretina.config import BACKBONES

    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(
    AVAILABLE,
    "CPU torch/torchvision locked extras required; dedicated CI executes these tests",
)
class Phase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_five_factories_forward_logits_no_download(self):
        for name in BACKBONES:
            with self.subTest(name=name):
                model = create_backbone(name).eval()
                with torch.inference_mode():
                    out = model(torch.zeros(1, 3, 64, 64))
                self.assertEqual(tuple(out.shape), (1, 8))
                self.assertTrue(torch.isfinite(out).all())
                self.assertEqual(model.provenance["initialization"], "random_unfitted")
                self.assertIsInstance(model.head[0], torch.nn.Dropout)
                del model

    def test_shared_geometry_and_evaluation_determinism(self):
        image = Image.fromarray(np.tile(np.arange(96, dtype=np.uint8), (48, 1)))
        config = PreprocessConfig(training_horizontal_flip=True)
        a, meta = preprocess(image, config, role="validation")
        b, _ = preprocess(image, config, role="validation")
        self.assertEqual(tuple(a.shape), (3, 224, 224))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(meta["horizontal_flip"])
        self.assertEqual(meta["resized_size"], [224, 112])
        with self.assertRaises(ValueError):
            preprocess(image, config, role="train")

    def test_true_multilabel_metrics_and_undefined_class(self):
        y = np.array([[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1]])
        report = multilabel_metrics(y, y * 0.8 + 0.1)
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["micro_f1"], 1.0)
        self.assertIsNone(
            multilabel_metrics(np.zeros((2, 8)), np.zeros((2, 8)))["per_class"]["H"][
                "auroc"
            ]
        )
        for wrong in (y.ravel(), y[:, :7], np.ones((1, 8)) * 2):
            with self.assertRaises(ValueError):
                multilabel_metrics(wrong, wrong)

    def test_patient_mean_before_threshold_and_label_consistency(self):
        y = np.ones((2, 8))
        p = np.array([[0.9] * 8, [0.3] * 8])
        r = aggregate_patient_mean(["p", "p"], ["left", "right"], y, p, ["g", "g"])
        np.testing.assert_allclose(r["probabilities"], [[0.6] * 8])
        self.assertEqual(r["labels"].shape, (1, 8))
        for eyes, labels, groups in [
            (["left", "left"], y, ["g", "g"]),
            (["left", "right"], np.array([[0] * 8, [1] * 8]), ["g", "g"]),
            (["left", "right"], y, ["a", "b"]),
        ]:
            with self.assertRaises(ValueError):
                aggregate_patient_mean(["p", "p"], eyes, labels, p, groups)

    def test_default_fitting_gates_before_weight_or_bn_mutation(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 8))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with self.assertRaises(ProtocolNotReady):
            train_epoch(model, [], optimizer)
        with self.assertRaises(ProtocolNotReady):
            fit_temperature(np.zeros((2, 8)), np.zeros((2, 8)), role="calibration_fit")
        with self.assertRaises(ProtocolNotReady):
            fit_reliability(role="calibration_policy")
        self.assertTrue(
            all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
        )
        self.assertFalse(optimizer.state)

    def test_fixed_temperature_transform_no_fit_and_review_default(self):
        logits = np.array([[1000.0, -1000.0, 0.0, 2.0, -2.0, 1.0, 3.0, -3.0]])
        p = TemperatureCalibration().probabilities(logits)
        self.assertEqual(p.shape, (1, 8))
        self.assertEqual(p[0, 0], 1.0)
        self.assertAlmostEqual(p[0, 2], 0.5)
        self.assertEqual(
            ReliabilityPolicy().decide(p, np.zeros_like(p), np.ones(1)).tolist(),
            ["REVIEW REQUIRED"],
        )
        with self.assertRaises(ValueError):
            TemperatureCalibration((0.0,))

    def test_mc_dropout_restores_modes_and_freezes_batchnorm(self):
        model = torch.nn.Sequential(
            torch.nn.BatchNorm1d(4), torch.nn.Dropout(0.5), torch.nn.Linear(4, 8)
        )
        model.train()
        before = model[0].running_mean.clone()
        flags = [m.training for m in model.modules()]
        result = mc_dropout(model, torch.ones(8, 4), passes=12)
        self.assertTrue(torch.equal(before, model[0].running_mean))
        self.assertEqual(flags, [m.training for m in model.modules()])
        self.assertGreater(float(result["variance"].sum()), 0.0)

    def test_checkpoint_untrained_state_rng_and_loader_restore(self):
        # No backward/optimizer step occurs. This checks serialization, not trained-run equivalence.
        model = torch.nn.Linear(3, 8)
        optimizer = torch.optim.AdamW(model.parameters())
        generator = torch.Generator().manual_seed(17)
        provenance = dict(
            backbone="synthetic_unfitted",
            split_sha256="a" * 64,
            config_sha256="b" * 64,
            preprocessing_sha256="c" * 64,
            seed=17,
            data_kind="synthetic",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            digest = save_checkpoint(
                path,
                model,
                optimizer,
                completed_epoch=2,
                global_step=0,
                provenance=provenance,
                generator=generator,
            )
            expected = (
                random.random(),
                np.random.random(),
                torch.rand(3),
                torch.randperm(12, generator=generator),
            )
            weights = model.weight.detach().clone()
            with torch.no_grad():
                model.weight.zero_()
            state = load_checkpoint(
                path,
                model,
                optimizer,
                expected_sha256=digest,
                expected_provenance=provenance,
                generator=generator,
            )
            self.assertTrue(torch.equal(weights, model.weight))
            self.assertEqual(state["next_epoch"], 3)
            self.assertEqual(expected[0], random.random())
            self.assertEqual(expected[1], np.random.random())
            self.assertTrue(torch.equal(expected[2], torch.rand(3)))
            self.assertTrue(
                torch.equal(expected[3], torch.randperm(12, generator=generator))
            )
            for checksum, prov in [
                ("0" * 64, provenance),
                (digest, dict(provenance, seed=99)),
            ]:
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        path,
                        model,
                        optimizer,
                        expected_sha256=checksum,
                        expected_provenance=prov,
                        generator=generator,
                    )
                self.assertTrue(torch.equal(weights, model.weight))

    def test_validation_selection_and_final_access_guards(self):
        reports = [
            dict(role="validation", epoch=2, macro_f1=0.7),
            dict(role="validation", epoch=1, macro_f1=0.7),
        ]
        self.assertEqual(select_checkpoint(reports)["epoch"], 1)
        self.assertEqual(
            select_backbone(
                {n: dict(role="validation", macro_f1=0.5) for n in BACKBONES}
            ),
            BACKBONES[0],
        )
        with self.assertRaises(ValueError):
            select_checkpoint(reports, role="final_evaluation")
        with self.assertRaises(ProtocolNotReady):
            evaluate(torch.nn.Linear(3, 8), [], role="final_evaluation")

    def test_bce_contract_and_read_only_manifest_adapter(self):
        self.assertAlmostEqual(
            float(training_loss(torch.zeros(2, 8), torch.zeros(2, 8))),
            np.log(2),
            places=6,
        )
        with self.assertRaises(ValueError):
            training_loss(torch.zeros(16), torch.zeros(16))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (48, 32), (80, 40, 20)).save(root / "eye.png")
            digest = sha256(root / "eye.png")
            rows = [
                dict(
                    patient_id="p",
                    eye=eye,
                    labels=[1, 0, 0, 0, 0, 0, 0, 0],
                    component_id="g",
                    path="eye.png",
                    raw_sha256=digest,
                )
                for eye in ("left", "right")
            ]
            manifest = root / "split.json"
            manifest.write_text(
                json.dumps(
                    dict(schema="phase2_eye_rows_v1", role="validation", rows=rows)
                )
            )
            data = PatientEyeDataset(
                manifest, expected_sha256=sha256(manifest), root=root, role="validation"
            )
            batch = collate_eyes([data[0], data[1]])
            self.assertEqual(tuple(batch["targets"].shape), (2, 8))
            self.assertEqual(batch["role"], "validation")
            Image.new("RGB", (48, 32)).save(root / "eye.png")
            with self.assertRaises(ValueError):
                data[0]

    def test_evaluator_aggregates_and_restores_training_mode(self):
        model = torch.nn.Linear(3, 8)
        model.train()
        batch = dict(
            role="validation",
            images=torch.zeros(2, 3),
            targets=torch.zeros(2, 8),
            patient_ids=["p", "p"],
            eyes=["left", "right"],
            component_ids=["g", "g"],
        )
        result = evaluate(model, [batch])
        self.assertEqual(result["per_patient"]["n"], 1)
        self.assertEqual(result["per_image"]["n"], 2)
        self.assertTrue(model.training)

    def test_duplicate_component_support_refuses_calibration_before_optimizer(self):
        from unittest.mock import patch
        from trustretina.training.protocol import ProtocolApproval

        approval = ProtocolApproval(
            True,
            "synthetic_contract",
            "a" * 64,
            True,
            None,
            (("calibration_fit", "b" * 64),),
        )
        with patch("torch.optim.LBFGS") as optimizer:
            with self.assertRaisesRegex(ProtocolNotReady, "independent per-class"):
                fit_temperature(
                    np.zeros((40, 8)),
                    np.zeros((40, 8)),
                    role="calibration_fit",
                    approval=approval,
                    allow_fitting=True,
                    patient_ids=[str(i) for i in range(40)],
                    component_ids=["same_copy_group"] * 40,
                    manifest_sha256="b" * 64,
                )
            optimizer.assert_not_called()
        self.assertEqual(TemperatureCalibration().source_role, "identity_unfitted")

    def test_component_bootstrap_is_deterministic_and_requires_independence(self):
        from trustretina.evaluation.metrics import component_bootstrap_f1

        y = np.array([[1] * 8, [1] * 8, [0] * 8, [0] * 8])
        p = y * 0.8 + 0.1
        a = component_bootstrap_f1(y, p, ["copy", "copy", "c", "d"], replicates=20)
        b = component_bootstrap_f1(y, p, ["copy", "copy", "c", "d"], replicates=20)
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            component_bootstrap_f1(y, p, ["copy"] * 4, replicates=20)
