import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from geolocalization.local_cache import FeatureCache, IsolatedRandomState, StageRecorder, ranked_candidates
from geolocalization.local_matching import LocalMatcher, PartialTransport, create_local_optimizer, matching_evidence
from geolocalization.local_retrieval import direction_losses, fixed_fusion_metrics, mine_local_candidates, rerank, score_candidates
from geolocalization.partial_experiment import train_local_epoch
from transformers import get_cosine_schedule_with_warmup
from sample4geo.model import TimmModel


def configuration():
    return json.loads((EXPERIMENT / "configs" / "geo_partial_match_gate.json").read_text(encoding="utf-8"))


def small_configuration():
    config = configuration()
    config.update(local_dimension=16, attention_heads=4, ffn_hidden=32,
                  pair_group_size=3, local_query_batch_size=3, rerank_candidates=6,
                  global_hard_count=1, local_hard_count=1, random_negative_count=1)
    return config


def feature_cache(view, count=10, context="same_global_checkpoint"):
    height, width = (2, 3) if view == "ground" else (2, 4)
    return FeatureCache(torch.arange(count), F.normalize(torch.randn(count, 8), dim=1),
                        torch.randn(count, 512, height, width), torch.randn(count, 1024, height, width), view, context)


class PartialMatchingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(4)

    def test_augmented_marginals_and_uniform_solution(self):
        transport = PartialTransport(0.1, 30, 500, 10, 1e-3)
        with torch.no_grad():
            transport.bin_score.zero_()
        real, diagnostics = transport(torch.zeros(2, 3, 8), torch.zeros(2, 5, 8))
        torch.testing.assert_close(real, torch.full_like(real, 1 / 8))
        self.assertTrue((diagnostics[:, :2] <= 1e-3).all())
        self.assertTrue((diagnostics[:, 2] == 30).all())
        self.assertTrue((real.sum(1) < 1).all())
        self.assertTrue((real.sum(2) < 1).all())

    def test_iteration_cap_and_nonfinite_rejection(self):
        transport = PartialTransport(0.01, 30, 40, 10, 1e-12)
        g = F.normalize(torch.randn(2, 5, 8), dim=-1).requires_grad_()
        s = F.normalize(torch.randn(2, 7, 8), dim=-1).requires_grad_()
        real, diagnostics = transport(g, s)
        self.assertTrue((diagnostics[:, 2] == 40).all())
        self.assertTrue(torch.isfinite(real).all())
        real.square().sum().backward()
        self.assertTrue(torch.isfinite(g.grad).all())
        with self.assertRaises(ValueError):
            transport(g.detach() * float("nan"), s.detach())

    def test_evidence_retains_matching_mass(self):
        g = F.normalize(torch.randn(2, 3, 16), dim=-1)
        s = F.normalize(torch.randn(2, 5, 16), dim=-1)
        weights = torch.full((2, 3, 5), 0.1)
        full, _ = matching_evidence(g, s, weights)
        half, _ = matching_evidence(g, s, weights / 2)
        torch.testing.assert_close(half[:, :32], full[:, :32] / 2)
        torch.testing.assert_close(half[:, 32], full[:, 32] / 2)

    def test_actual_dimensions_and_shared_encoder_gradient_boundary(self):
        config = configuration()
        config["activation_checkpointing"] = False
        model = LocalMatcher(config)
        g2 = torch.randn(1, 512, 4, 24, requires_grad=True)
        g3 = torch.randn(1, 1024, 4, 24, requires_grad=True)
        s2 = torch.randn(1, 512, 12, 12, requires_grad=True)
        s3 = torch.randn(1, 1024, 12, 12, requires_grad=True)
        ground, satellite = model.encode(g2, g3, "ground"), model.encode(s2, s3, "satellite")
        self.assertEqual(ground.shape, (1, 96, 128))
        self.assertEqual(satellite.shape, (1, 144, 128))
        scores, stats, diagnostics = model.score_pairs(ground, satellite)
        (scores.sum() + stats.sum() * 0.01).backward()
        for value in (g2, g3, s2, s3):
            self.assertIsNone(value.grad)
        for module in (model.projection2, model.projection3, model.scoring, model.transport):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertEqual(diagnostics.shape, (1, 4))

    def test_checkpoint_and_pair_group_gradient_equivalence(self):
        config = small_configuration()
        direct = LocalMatcher(config)
        direct.checkpointing = False
        grouped = copy.deepcopy(direct)
        grouped.checkpointing = True
        grouped.pair_group_size = 2
        g2, g3 = torch.randn(5, 512, 2, 3), torch.randn(5, 1024, 2, 3)
        s2, s3 = torch.randn(5, 512, 2, 4), torch.randn(5, 1024, 2, 4)
        outputs = []
        for model in (direct, grouped):
            scores, _, _ = model.score_pairs(model.encode(g2, g3, "ground"), model.encode(s2, s3, "satellite"))
            scores.square().sum().backward()
            outputs.append(scores.detach())
        torch.testing.assert_close(*outputs, rtol=1e-4, atol=1e-5)
        for (name, left), (_, right) in zip(direct.named_parameters(), grouped.named_parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=2e-3, atol=2e-5, msg=name)

    def test_query_gate_detaches_global_scores_and_preserves_consensus(self):
        model = LocalMatcher(small_configuration())
        global_scores = torch.tensor([[0.8, 0.7, 0.2], [0.3, 0.7, 0.2]], requires_grad=True)
        local_scores = torch.tensor([[0.7, 0.6, 0.1], [0.5, 0.8, 0.4]], requires_grad=True)
        fused, beta = model.fuse(global_scores, local_scores, torch.zeros(2, 3, 6))
        torch.testing.assert_close(beta, torch.full_like(beta, 0.05))
        self.assertEqual(fused.argmax(1).tolist(), global_scores.argmax(1).tolist())
        F.cross_entropy(fused, torch.tensor([0, 1])).backward()
        self.assertIsNone(global_scores.grad)
        self.assertGreater(float(local_scores.grad.abs().sum()), 0)
        self.assertIsNotNone(model.fusion_logit.grad)

    def test_original_convnext_forward_with_read_only_cache_hooks(self):
        model = TimmModel("convnext_base.fb_in22k_ft_in1k_384", pretrained=False, img_size=384).eval()
        images = torch.randn(1, 3, 140, 768)
        with torch.no_grad():
            expected = model(images)
            state = torch.random.get_rng_state().clone()
            with StageRecorder(model, "ground") as recorder:
                actual, stages = recorder.forward(images)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(stages[0].shape, (1, 512, 4, 24))
        self.assertEqual(stages[1].shape, (1, 1024, 4, 24))
        self.assertEqual(model.model.head.norm.eps, 1e-6)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertFalse(model.model.stages[2]._forward_hooks)

    def test_random_state_and_new_module_initialization_isolation(self):
        python = random.getstate()
        numpy = np.random.get_state()
        state = torch.random.get_rng_state().clone()
        with IsolatedRandomState():
            random.random()
            np.random.random()
            LocalMatcher(small_configuration())
        self.assertEqual(python, random.getstate())
        self.assertTrue(np.array_equal(numpy[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_negative_lists_losses_optimizer_and_state_reload(self):
        config = small_configuration()
        model = LocalMatcher(config)
        model.checkpointing = False
        ground, satellite = feature_cache("ground"), feature_cache("satellite")
        global_scores = ground.global_features @ satellite.global_features.T
        lists, top = mine_local_candidates(model, ground, satellite, global_scores, config, "cpu", 42)
        self.assertEqual(lists.shape, (10, 4))
        self.assertEqual(lists[:, 0].tolist(), list(range(10)))
        for row in lists.tolist():
            self.assertEqual(len(set(row)), 4)
        optimizer = create_local_optimizer(model, config)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        before = model.projection2.weight.detach().clone()
        local, fusion, info = direction_losses(model, ground, satellite, [0, 1, 2], lists, top,
                                               global_scores, torch.tensor(5.0), config, "cpu")
        (0.5 * local + 0.5 * fusion).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.projection2.weight))
        self.assertEqual(info["queries"], 3)
        state = copy.deepcopy(model.state_dict())
        expected = rerank(model, ground, satellite, global_scores, config, "cpu")
        loaded = LocalMatcher(config)
        loaded.load_state_dict(state, strict=True)
        actual = rerank(loaded, ground, satellite, global_scores, config, "cpu")
        self.assertEqual(actual["ranks"], expected["ranks"])
        self.assertEqual(actual["top1_ids"], expected["top1_ids"])
        fixed = rerank(loaded, ground, satellite, global_scores, config, "cpu", fixed_beta=0.4)
        self.assertEqual(fixed_fusion_metrics(actual, ground, satellite, global_scores, 0.4), fixed["metrics"])

    def test_wrong_cache_context_rejected_and_tie_order_is_stable(self):
        scores = torch.ones(2, 4)
        ids = torch.tensor([8, 2, 5, 1])
        self.assertEqual(ranked_candidates(scores, ids, 3).tolist(), [[3, 1, 2], [3, 1, 2]])
        with self.assertRaises(ValueError):
            FeatureCache(torch.tensor([1, 1]), torch.ones(2, 3), torch.ones(2, 512, 2, 2),
                         torch.ones(2, 1024, 2, 2), "ground", "checkpoint")
        with self.assertRaises(ValueError):
            score_candidates(LocalMatcher(small_configuration()), feature_cache("ground", context="first"),
                             feature_cache("satellite", context="second"), [0], [[0, 1]], "cpu")

    def test_local_training_epoch_and_sampling_audit(self):
        config = small_configuration()
        config["mixed_precision"] = False
        model = LocalMatcher(config)
        model.checkpointing = False
        ground, satellite = feature_cache("ground"), feature_cache("satellite")
        scores = ground.global_features @ satellite.global_features.T
        optimizer = create_local_optimizer(model, config)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 1, 3)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        root = EXPERIMENT.parents[1] / ".local-cache"
        with tempfile.TemporaryDirectory(dir=root, prefix="partial-check-") as directory:
            output = Path(directory).resolve()
            self.assertTrue(output.is_relative_to(root.resolve()))
            with IsolatedRandomState():
                rows, step = train_local_epoch(model, (ground, satellite), {"scores": scores},
                    [0, 1, 2, 3, 4, 5], torch.tensor(5.0), config, optimizer, scheduler, scaler,
                    epoch=1, step=0, output=output)
            self.assertEqual(step, 2)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["learning_rates"]["local"], 0)
            self.assertGreater(rows[1]["learning_rates"]["local"], 0)
            audit = json.loads((output / "local_candidates_epoch_1.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["query_ids"], [0, 1, 2, 3, 4, 5])
            self.assertEqual(len(audit["local_parameter_sha256"]), 64)
            self.assertTrue(all(torch.isfinite(torch.tensor(row["loss"])) for row in rows))

    def test_local_loss_still_trains_when_positive_is_outside_fusion_candidates(self):
        config = small_configuration()
        model = LocalMatcher(config)
        model.checkpointing = False
        ground, satellite = feature_cache("ground"), feature_cache("satellite")
        local_lists = torch.tensor([[index, (index + 1) % 10, (index + 2) % 10, (index + 3) % 10]
                                    for index in range(10)])
        fusion_lists = torch.tensor([[(index + offset) % 10 for offset in range(1, 7)] for index in range(10)])
        scores = ground.global_features @ satellite.global_features.T
        local, fusion, info = direction_losses(model, ground, satellite, [0, 1], local_lists, fusion_lists,
                                               scores, torch.tensor(5.0), config, "cpu")
        self.assertEqual(float(fusion.detach()), 0)
        self.assertEqual(info["valid_fusion_queries"], 0)
        local.backward()
        self.assertGreater(float(model.projection3.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
