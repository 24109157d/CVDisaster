import copy
import json
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from geolocalization.loss import DescriptorInfoNCE
from geolocalization.model import GeoModel
from geolocalization.runtime import GeoDataParallel, create_optimizer
from geolocalization.spatial import FeatureTransport, create_spatial_modules, transport_statistics
from sample4geo.loss import InfoNCE


class TransportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_early_stop_and_backward(self):
        projection, transport, refinement = create_spatial_modules(1.0, 30, 42)
        source = torch.randn(3, 1024, 4, 16, requires_grad=True)
        moved, matrix, diagnostics = transport(projection(source), torch.arange(3))
        self.assertEqual(diagnostics.shape, (3, 6))
        self.assertTrue((diagnostics[:, 4] == 30).all())
        self.assertTrue((diagnostics[:, 5] == 1).all())
        self.assertLessEqual(float((matrix.sum(2) - 1).abs().max()), 1e-3)
        refinement(moved).square().mean().backward()
        for module in (projection, transport, refinement):
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(source.grad.abs().max()), 0)

    def test_iteration_cap_continues_with_finite_matrix(self):
        transport = FeatureTransport(max_iterations=500, tolerance=1e-12)
        with torch.no_grad():
            transport.cost.bias.copy_(torch.randn(4096) * 20)
        source = torch.randn(2, 64, 4, 16, requires_grad=True)
        moved, matrix, diagnostics = transport(source, torch.arange(2))
        self.assertTrue((diagnostics[:, 4] == 500).all())
        self.assertTrue((diagnostics[:, 5] == 0).any())
        self.assertTrue(torch.isfinite(matrix).all())
        moved.square().mean().backward()
        self.assertTrue(torch.isfinite(source.grad).all())
        report = transport_statistics(diagnostics)
        self.assertEqual(report["max_iterations"], 500)
        self.assertGreater(report["unconverged_samples"], 0)

    def test_checks_continue_on_ten_iteration_boundaries(self):
        transport = FeatureTransport(tolerance=1e-3)
        with torch.no_grad():
            transport.cost.bias.copy_(torch.randn(4096) * 10)
        _, matrix, diagnostics = transport(torch.randn(3, 64, 4, 16), torch.arange(3))
        iterations = int(diagnostics[0, 4])
        self.assertGreater(iterations, 30)
        self.assertEqual((iterations - 30) % 10, 0)
        self.assertLessEqual(iterations, 500)
        if iterations < 500:
            self.assertTrue((diagnostics[:, 5] == 1).all())
        torch.testing.assert_close(diagnostics[:, 2], (matrix.sum(2) - 1).abs().amax(1))

    def test_nonfinite_cost_fails(self):
        source = torch.randn(1, 64, 4, 16)
        source[0, 0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            FeatureTransport()(source, torch.tensor([1]))

    def test_initialization_preserves_rng(self):
        before = torch.get_rng_state().clone()
        create_spatial_modules(1.0, 30, 42)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.original = InfoNCE(nn.CrossEntropyLoss(label_smoothing=0.1), device="cpu")

    def test_baseline_loss_and_gradients_match_original(self):
        first = torch.randn(7, 11, requires_grad=True)
        second = torch.randn(7, 11, requires_grad=True)
        scale = torch.tensor(2.0, requires_grad=True)
        actual = DescriptorInfoNCE(nn.CrossEntropyLoss(label_smoothing=0.1), device="cpu")(
            {"descriptors": first}, {"descriptors": second}, scale)
        expected = self.original(first, second, scale)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for left, right in zip(torch.autograd.grad(actual, (first, second, scale), retain_graph=True),
                               torch.autograd.grad(expected, (first, second, scale))):
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_three_losses_and_all_gradients(self):
        first_global = torch.randn(7, 11, requires_grad=True)
        second_global = torch.randn(7, 11, requires_grad=True)
        first_spatial = torch.randn(7, 13, requires_grad=True)
        second_spatial = torch.randn(7, 13, requires_grad=True)
        fusion_logit = torch.tensor(math.log(7 / 3), requires_grad=True)
        scale = torch.tensor(2.0, requires_grad=True)
        alpha = fusion_logit.sigmoid()

        def outputs(global_raw, spatial_raw):
            spatial = F.normalize(spatial_raw, dim=1)
            descriptor = torch.cat((alpha.sqrt() * F.normalize(global_raw, dim=1),
                                    (1 - alpha).sqrt() * spatial), dim=1)
            return {"global_raw": global_raw, "spatial_features": spatial, "descriptors": descriptor}

        first, second = outputs(first_global, first_spatial), outputs(second_global, second_spatial)
        criterion = DescriptorInfoNCE(nn.CrossEntropyLoss(label_smoothing=0.1), device="cpu")
        actual = criterion(first, second, scale)
        expected = sum(self.original(first[key], second[key], scale)
                       for key in ("global_raw", "spatial_features", "descriptors"))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        inputs = (first_global, second_global, first_spatial, second_spatial, fusion_logit, scale)
        for left, right in zip(torch.autograd.grad(actual, inputs, retain_graph=True),
                               torch.autograd.grad(expected, inputs)):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
            self.assertTrue(torch.isfinite(left).all())
            self.assertGreater(float(left.abs().max()), 0)
        self.assertEqual(set(criterion.components), {"global", "spatial", "fusion"})


class ModelConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((EXPERIMENT / "configs/geo_cvft_refine.json").read_text(encoding="utf-8"))
        torch.manual_seed(42)
        cls.model = GeoModel("geo_cvft_refine")

    def test_original_parameters_and_rng_match_baseline(self):
        torch.manual_seed(42)
        baseline = GeoModel("baseline")
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, self.model.state_dict()[name], atol=0, rtol=0)

    def test_optimizer_preserves_geo_and_covers_all_parameters_once(self):
        optimizer = create_optimizer(self.model, self.config)
        groups = {group["name"]: group for group in optimizer.param_groups}
        self.assertEqual({name: group["lr"] for name, group in groups.items()},
                         {"geo": 5e-4, "spatial": 5e-4, "fusion": 1e-2})
        self.assertEqual(groups["geo"]["weight_decay"], 0.01)
        self.assertEqual(groups["spatial"]["weight_decay"], 0.01)
        self.assertEqual(groups["fusion"]["weight_decay"], 0)
        self.assertIn(id(self.model.logit_scale), {id(value) for value in groups["geo"]["params"]})
        identifiers = [id(value) for group in optimizer.param_groups for value in group["params"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(set(identifiers), {id(value) for value in self.model.parameters()})

    def test_global_optimizer_update_matches_original(self):
        self.model.zero_grad(set_to_none=True)
        optimizer = create_optimizer(self.model, self.config)
        parameter = self.model.logit_scale
        saved = parameter.detach().clone()
        reference = nn.Parameter(saved.clone())
        expected = torch.optim.AdamW([reference], lr=self.config["lr"], betas=tuple(self.config["betas"]),
                                     eps=self.config["eps"], weight_decay=self.config["weight_decay"])
        parameter.grad = torch.tensor(0.25)
        reference.grad = parameter.grad.clone()
        optimizer.step()
        expected.step()
        torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
        with torch.no_grad():
            parameter.copy_(saved)
        optimizer.zero_grad(set_to_none=True)

    def test_fusion_initialization_and_checkpoint(self):
        self.assertAlmostEqual(self.model.fusion_state()["fusion_weight"], 0.7, places=6)
        alpha, global_scale, spatial_scale = self.model.fusion_coefficients()
        torch.testing.assert_close(global_scale.square(), alpha)
        torch.testing.assert_close(spatial_scale.square(), 1 - alpha)
        state = copy.deepcopy(self.model.state_dict())
        with torch.no_grad():
            self.model.fusion_logit.add_(1)
        self.model.load_state_dict(state, strict=True)
        self.assertAlmostEqual(self.model.fusion_state()["fusion_weight"], 0.7, places=6)

    def test_gathered_spatial_completion_and_gradient(self):
        for view in ("ground", "satellite"):
            self.model.zero_grad(set_to_none=True)
            raw = torch.randn(3, 1024, requires_grad=True)
            source = torch.randn(3, 64, 4, 16) if view == "ground" else torch.randn(3, 64, 8, 8)
            source.requires_grad_()
            encoded = {"global_raw": raw, "global_features": F.normalize(raw, dim=1),
                       "spatial_source": source}
            result = self.model.complete_encoding(encoded, view, torch.arange(3))
            self.assertEqual(result["descriptors"].shape, (3, 5120))
            (result["descriptors"] * torch.randn_like(result["descriptors"])).sum().backward()
            self.assertIsNotNone(raw.grad)
            self.assertIsNotNone(source.grad)
            self.assertTrue(torch.isfinite(source.grad).all())
            self.assertGreater(float(source.grad.abs().max()), 0)
            if view == "ground":
                self.assertEqual(len(result["transport_diagnostics"][:, 4].unique()), 1)

    def test_parallel_forward_and_direct_forward_match(self):
        self.model.eval()
        ground = torch.randn(1, 3, 140, 768)
        satellite = torch.randn(1, 3, 384, 384)
        ids = torch.tensor([123])
        parallel = GeoDataParallel(self.model, device_ids=[])
        with torch.no_grad():
            direct = self.model(ground, satellite, sample_ids=ids)
            gathered = parallel(ground, satellite, sample_ids=ids)
        for first, second in zip(direct, gathered):
            torch.testing.assert_close(first["descriptors"], second["descriptors"], atol=0, rtol=0)
            torch.testing.assert_close(first["global_raw"], second["global_raw"], atol=0, rtol=0)

    def test_complete_training_step_updates_all_module_groups(self):
        self.model.train()
        optimizer = create_optimizer(self.model, self.config)
        optimizer.zero_grad(set_to_none=True)
        parameters = {"backbone": self.model.model.stem[0].weight,
                      "projection": self.model.channel_projection.weight,
                      "transport": self.model.feature_transport.cost.weight,
                      "refinement": self.model.refinement.conv2.weight,
                      "fusion": self.model.fusion_logit, "scale": self.model.logit_scale}
        before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        ground = torch.randn(2, 3, 140, 768)
        satellite = torch.randn(2, 3, 384, 384)
        parallel = GeoDataParallel(self.model, device_ids=[])
        first, second = parallel(ground, satellite, sample_ids=torch.arange(2))
        criterion = DescriptorInfoNCE(nn.CrossEntropyLoss(label_smoothing=0.1), device="cpu",
                                     weights=self.config["loss_weights"])
        loss = criterion(first, second, self.model.logit_scale.exp())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in parameters.items():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().max()), 0, name)
        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config["clip_grad"])
        optimizer.step()
        for name, parameter in parameters.items():
            self.assertFalse(torch.equal(before[name], parameter), name)
        self.model.zero_grad(set_to_none=True)
        # 后续检查继续使用同一份初始参数。
        torch.manual_seed(42)
        type(self).model = GeoModel("geo_cvft_refine")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
