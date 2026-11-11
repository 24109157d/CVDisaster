import argparse
import gc
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from geolocalization.author_validation import validate_author_evaluation
from geolocalization.data import make_datasets, make_loaders
from geolocalization.evaluation import (evaluate, feature_context, load_candidate_cache,
                                       save_candidate_cache)
from geolocalization.loss import contrastive_loss
from geolocalization.model import normalize_features
from geolocalization.runtime import (environment, initialize, load_initial_model, memory_usage,
                                     new_output, parallel_model, read_config, require_compute_node,
                                     sha256, write_json)
from geolocalization.training import check_gradients
from sample4geo.loss import InfoNCE


def check_fusion_outputs(model, ground, satellite):
    with torch.no_grad(), autocast(enabled=False):
        alpha, global_scale, spatial_scale = model.module.fusion_coefficients()
        torch.testing.assert_close(global_scale.square(), alpha, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(spatial_scale.square(), 1 - alpha, atol=1e-7, rtol=1e-6)
        raw = []
        for result in (ground, satellite):
            unnormalized = torch.cat((global_scale * result["global_features"].float(),
                                      spatial_scale * result["spatial_features"].float()), dim=1)
            for name in ("global_features", "spatial_features", "descriptors"):
                values = result[name]
                if values.dtype != torch.float32 or not torch.isfinite(values).all():
                    raise AssertionError(f"{name} 的 dtype 或数值错误")
                torch.testing.assert_close(values.norm(dim=1), torch.ones_like(values[:, 0]), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(unnormalized.norm(dim=1), torch.ones_like(unnormalized[:, 0]), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(result["descriptors"], normalize_features(unnormalized, "fusion_validation"),
                                       atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(result["fusion_weight"], alpha.expand(len(unnormalized)), rtol=0, atol=0)
            raw.append(unnormalized)
        weighted = (alpha * (ground["global_features"] @ satellite["global_features"].T)
                    + (1 - alpha) * (ground["spatial_features"] @ satellite["spatial_features"].T))
        torch.testing.assert_close(raw[0] @ raw[1].T, weighted, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ground["descriptors"] @ satellite["descriptors"].T,
                                   weighted, atol=1e-5, rtol=1e-5)
    return model.module.fusion_state()


def validate_architecture(config, architecture, output):
    initialize(42)
    model, scales = load_initial_model(config, architecture)
    datasets, manifest, transforms = make_datasets(config, model.get_config())
    loaders = make_loaders(datasets, config, 42)
    model = parallel_model(model)
    initial_fusion = model.module.fusion_state()
    if architecture == "geo_cvft_refine" and (initial_fusion["fusion_logit"], initial_fusion["fusion_weight"]) != (0.0, 0.5):
        raise AssertionError("初始融合参数必须为 0，对应比例为 0.5")
    for index in range(2):
        torch.cuda.reset_peak_memory_stats(index)
    ground_image, satellite_image, sample_ids = next(iter(loaders["train"]))
    ground_image, satellite_image, sample_ids = ground_image.cuda(), satellite_image.cuda(), sample_ids.cuda()
    if ground_image.shape != (64, 3, 140, 768) or satellite_image.shape != (64, 3, 384, 384):
        raise ValueError("真实训练 batch 的形状不符合方案")
    model.eval()
    with torch.no_grad(), autocast():
        paired_ground, paired_satellite = model(ground_image, satellite_image, sample_ids=sample_ids)
        single_ground = model(ground_image, view="ground", sample_ids=sample_ids)
        single_satellite = model(satellite_image, view="satellite", sample_ids=sample_ids)
        torch.testing.assert_close(paired_ground["descriptors"], single_ground["descriptors"])
        torch.testing.assert_close(paired_satellite["descriptors"], single_satellite["descriptors"])
        # 重新排列卫星 batch，街景矩阵与向量应保持原值。
        independent_ground, _ = model(ground_image, satellite_image.flip(0), sample_ids=sample_ids)
        torch.testing.assert_close(paired_ground["descriptors"], independent_ground["descriptors"])
        if architecture == "geo_cvft_refine":
            torch.testing.assert_close(paired_ground["transport"], independent_ground["transport"])
            if paired_ground["transport"].dtype != torch.float32:
                raise TypeError("AMP 中 Sinkhorn 必须使用 FP32")
    if architecture == "geo_cvft_refine":
        check_fusion_outputs(model, paired_ground, paired_satellite)
        with torch.no_grad(), autocast():
            raw = model.module.model.forward_features(ground_image[:2])
            global_expected = model.module.model.forward_head(raw).float()
            source_map = model.module.ground_pool(model.module.channel_projection(raw))
        with torch.no_grad(), autocast(enabled=False):
            moved, transport, _ = model.module.feature_transport(source_map, sample_ids[:2])
            source = source_map.float().flatten(2).transpose(1, 2)
            expected = torch.einsum("bij,bic->bjc", transport, source)
            actual = moved.flatten(2).transpose(1, 2)
            torch.testing.assert_close(actual, expected)
            error = (actual.mean(1) - source.mean(1)).abs().amax(1)
            # 有限次 Sinkhorn 的均值误差由源特征幅度和行和误差共同限定。
            row_error = (transport.sum(2) - 1).abs().amax(1)
            bound = row_error * source.abs().amax((1, 2)) + 1e-5
            if (error > bound).any():
                raise AssertionError(f"搬移均值关系错误: error={error.tolist()}, bound={bound.tolist()}")
            flat = moved.flatten(2).transpose(1, 2).reshape(2, 4096)
            restored = flat.reshape(2, 64, 64).transpose(1, 2).reshape(2, 64, 8, 8)
            torch.testing.assert_close(restored, moved)
            normalized_expected = global_expected / global_expected.norm(dim=1, keepdim=True)
            # DataParallel 每张卡的 batch 规模不同，允许 CUDA 算子造成的数值误差。
            torch.testing.assert_close(normalized_expected, paired_ground["global_features"][:2],
                                       atol=2e-3, rtol=2e-3)
    del paired_ground, paired_satellite, single_ground, single_satellite, independent_ground
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]),
                                  eps=config["eps"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
    scaler = GradScaler(init_scale=config["grad_scaler_init"])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    before = {"model": model.module.model.stem[0].weight.detach().clone(),
              "logit_scale": model.module.logit_scale.detach().clone()}
    if architecture == "geo_cvft_refine":
        before.update(channel_projection=model.module.channel_projection.weight.detach().clone(),
                      feature_transport=model.module.feature_transport.cost.weight.detach().clone(),
                      refinement=model.module.refinement.conv2.weight.detach().clone(),
                      fusion_logit=model.module.fusion_logit.detach().clone())
    with autocast():
        ground, satellite = model(ground_image, satellite_image, sample_ids=sample_ids)
    loss = contrastive_loss(ground, satellite, model.module, sample_ids, config["label_smoothing"])
    with torch.no_grad(), autocast(enabled=False):
        expected_loss = InfoNCE(nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"], reduction="mean"),
                                device=sample_ids.device)(ground["descriptors"], satellite["descriptors"],
                                                         model.module.logit_scale.exp())
        torch.testing.assert_close(loss, expected_loss, atol=1e-5, rtol=1e-5)
    fusion_loss_gradient = (torch.autograd.grad(loss, model.module.fusion_logit, retain_graph=True)[0].detach()
                            if architecture == "geo_cvft_refine" else None)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = check_gradients(model.module)
    if architecture == "geo_cvft_refine":
        torch.testing.assert_close(model.module.fusion_logit.grad, fusion_loss_gradient, atol=1e-5, rtol=1e-5)
    torch.nn.utils.clip_grad_value_(model.parameters(), config["clip_grad"])
    scaler.step(optimizer)
    scaler.update()
    after = {"model": model.module.model.stem[0].weight, "logit_scale": model.module.logit_scale}
    if architecture == "geo_cvft_refine":
        after.update(channel_projection=model.module.channel_projection.weight,
                     feature_transport=model.module.feature_transport.cost.weight,
                     refinement=model.module.refinement.conv2.weight, fusion_logit=model.module.fusion_logit)
    for name, previous in before.items():
        if torch.equal(previous, after[name]) or not torch.isfinite(after[name]).all():
            raise AssertionError(f"真实 batch 更新后参数未改变或数值错误: {name}")
    model.eval()
    if architecture == "geo_cvft_refine":
        with torch.no_grad(), autocast():
            updated_ground, updated_satellite = model(ground_image, satellite_image, sample_ids=sample_ids)
        check_fusion_outputs(model, updated_ground, updated_satellite)
        del updated_ground, updated_satellite
    final_fusion = model.module.fusion_state()
    del ground, satellite, ground_image, satellite_image, before, after
    # 完整 128 幅评价 batch 与 1649 幅候选库验证实际显存和查询 ID 关联。
    checkpoint_path = output / f"{architecture}_validation.pth"
    torch.save({"model": model.module.state_dict(), "fusion": final_fusion}, checkpoint_path)
    checkpoint_id = sha256(checkpoint_path)
    result = evaluate(model, loaders, checkpoint_id=checkpoint_id)
    author_consistency = (validate_author_evaluation(checkpoint_path, loaders, result, config, output)
                          if architecture == "baseline" else None)
    cache_path = output / f"{architecture}_satellite_features.pt"
    save_candidate_cache(cache_path, result["satellite"], result["satellite_ids"], result["feature_context"])
    state = torch.load(checkpoint_path, map_location="cpu")
    model.module.load_state_dict(state["model"], strict=True)
    if model.module.fusion_state() != state["fusion"]:
        raise AssertionError("checkpoint 内的融合参数与记录不一致")
    del state
    if model.module.fusion_state() != final_fusion:
        raise AssertionError("checkpoint 加载后的融合参数与保存前不一致")
    cache = load_candidate_cache(cache_path, feature_context(model, checkpoint_id))
    torch.testing.assert_close(cache["features"]["descriptors"], result["satellite"]["descriptors"], atol=0, rtol=0)
    if not torch.equal(cache["ids"], result["satellite_ids"]):
        raise AssertionError("候选缓存的 ID 顺序不一致")
    reloaded = evaluate(model, loaders, checkpoint_id=checkpoint_id)
    for view in ("ground", "satellite"):
        torch.testing.assert_close(reloaded[view]["descriptors"], result[view]["descriptors"], atol=1e-5, rtol=1e-5)
        if not torch.equal(reloaded[view + "_ids"], result[view + "_ids"]):
            raise AssertionError(f"checkpoint 加载后 {view} 的 ID 顺序改变")
    if not torch.equal(reloaded["ranks"], result["ranks"]) or reloaded["metrics"] != result["metrics"]:
        raise AssertionError("checkpoint 加载后的完整测试集排名或 Recall 不一致")
    report = {"architecture": architecture, "initial_logit_scales": scales,
              "split_manifest": manifest, "transforms": transforms, "gradients": gradients,
              "loss": float(loss.detach()),
              "loss_input": "descriptors", "loss_logits_shape": [len(sample_ids), len(sample_ids)],
              "initial_fusion": initial_fusion, "final_fusion": final_fusion,
              "fusion_logit_gradient": float(fusion_loss_gradient) if fusion_loss_gradient is not None else None,
              "author_evaluation_consistency": author_consistency,
              "feature_context": result["feature_context"], "candidate_cache_verified": True,
              "metrics": result["metrics"], "queries": len(result["ranks"]), "training_batch": 64,
              "evaluation_batch": 128, "timing": result["timing"], "memory": memory_usage(),
              "checkpoint_reload_verified": True, "passed": True}
    write_json(output / f"{architecture}_validation.json", report)
    del model, optimizer, scaler, loss, expected_loss
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description="使用真实 CVIAN 数据和两张 A100 验收模型")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs/geo_cvft_refine.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require_compute_node()
    config = read_config(args.config)
    output = new_output(args.output)
    reports = [validate_architecture(config, architecture, output) for architecture in config["architectures"]]
    write_json(output / "validation.json", {"environment": environment(), "models": reports, "passed": True})


if __name__ == "__main__":
    main()
