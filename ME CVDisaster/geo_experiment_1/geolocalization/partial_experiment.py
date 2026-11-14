import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import torch
import timm
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

from . import REPOSITORY
from .data import make_datasets, make_loaders, shuffle_training
from .local_cache import IsolatedRandomState, predict_cached
from .local_matching import LocalMatcher, create_local_optimizer
from .local_retrieval import direction_losses, fixed_fusion_metrics, mine_local_candidates, rerank
from .runtime import environment, initialize, new_output, require_compute_node, sha256, write_json
from sample4geo.evaluate.cvusa_and_cvact import calculate_nearest, calculate_scores
from sample4geo.loss import InfoNCE
from sample4geo.model import TimmModel
from sample4geo.trainer import train as author_train


def reproduction_configuration(config):
    if torch.__version__ != "1.13.1+cu116" or timm.__version__ != "0.8.19dev0":
        raise ValueError("正式运行需要复现环境的 PyTorch 1.13.1+cu116 与 timm 0.8.19dev0")
    path = REPOSITORY / "Author CVDisaster" / "geolocalization" / "1_train_cvdisaster.py"
    spec = importlib.util.spec_from_file_location("geo_reproduction_configuration", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original = module.Configuration()
    expected = {"model": "convnext_base.fb_in22k_ft_in1k_384", "img_size": 384,
                "split_key": "val60", "scheduler": "cosine", "custom_sampling": True,
                "sim_sample": True, "gps_sample": False, "normalize_features": True,
                "decay_exclue_bias": False, "grad_checkpointing": False, "zero_shot": True,
                "gpu_ids": (0, 1)}
    for key in ("seed", "epochs", "batch_size", "batch_size_eval", "num_workers", "lr", "label_smoothing",
                "clip_grad", "warmup_epochs", "mixed_precision", "prob_flip", "prob_rotate",
                "neighbour_select", "neighbour_range"):
        expected[key] = config[key]
    for name, value in expected.items():
        if getattr(original, name) != value:
            raise ValueError(f"原 geo 配置与新实验声明不一致: {name}")
    if Path(original.data_folder).resolve() != Path(config["data_root"]).resolve():
        raise ValueError("原 geo 数据目录与新实验不一致")
    if config["scheme"] != "geo_partial_match_gate" or config["local_query_batch_size"] != original.batch_size:
        raise ValueError("实验编号或新增模块查询 batch 错误")
    if config["rerank_candidates"] < 2 + config["global_hard_count"] + config["local_hard_count"]:
        raise ValueError("局部候选范围不足")
    if any(value <= 0 for value in config["loss_weights"].values()):
        raise ValueError("辅助损失系数必须为正数")
    if any(config[key] < 0 for key in ("global_hard_count", "local_hard_count", "random_negative_count")):
        raise ValueError("负候选数量不得为负数")
    if config["betas"] != [0.9, 0.999] or config["eps"] != 1e-8 or config["weight_decay"] != 0.01:
        raise ValueError("原 geo 的 AdamW 参数必须保持复现值")
    if set(config["loss_weights"]) != {"local", "fusion"}:
        raise ValueError("新增模块损失必须包含 local 和 fusion")
    return original


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cached_global_evaluation(original, model, loaders, split, context, config):
    context = context + ":" + split
    satellite, satellite_features, satellite_ids = predict_cached(
        original, model, loaders[split + "_satellite"], "satellite", context)
    ground, ground_features, ground_ids = predict_cached(
        original, model, loaders[split + "_ground"], "ground", context)
    result = calculate_scores(ground_features, satellite_features, ground_ids, satellite_ids,
                              step_size=1000, ranks=[1, 5, 10], return_details=True)
    neighbours = None
    if split == "train":
        neighbours = calculate_nearest(ground_features, satellite_features, ground_ids, satellite_ids,
                                       neighbour_range=config["neighbour_range"], step_size=1000)
    return ground, satellite, result, neighbours


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_checkpoint(path, global_model, local_model, config, manifest, epoch, global_step, local_step, metrics,
                    global_only=False):
    saved = {"architecture": "geo_partial_match_gate", "config": config, "split_manifest": manifest,
             "epoch": epoch, "global_step": global_step, "local_step": local_step,
             "global": cpu_state(global_model.module), "metrics": metrics, "global_only": global_only}
    if not global_only:
        saved["local"] = cpu_state(local_model)
    torch.save(saved, path)


def train_local_epoch(model, caches, global_result, sample_ids, scale, config, optimizer, scheduler, scaler,
                      epoch, step, output):
    device = next(model.parameters()).device
    ground, satellite = caches
    scores = global_result["scores"]
    g_lists, g_top = mine_local_candidates(model, ground, satellite, scores, config, device, config["seed"] + 2 * epoch)
    s_lists, s_top = mine_local_candidates(model, satellite, ground, scores.T, config, device, config["seed"] + 2 * epoch + 1)
    ground_indices = ground.indices(sample_ids)
    satellite_indices = satellite.indices(sample_ids)
    write_json(output / f"local_candidates_epoch_{epoch}.json", {
        "context": ground.context, "query_ids": sample_ids,
        "local_parameter_sha256": model_fingerprint(model),
        "ground_candidates": satellite.ids[g_lists[ground_indices]].tolist(),
        "satellite_candidates": ground.ids[s_lists[satellite_indices]].tolist()})
    model.train()
    rows = []
    for start in range(0, len(sample_ids), config["local_query_batch_size"]):
        stop = min(start + config["local_query_batch_size"], len(sample_ids))
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=config["mixed_precision"]):
            g_local, g_fusion, g_info = direction_losses(model, ground, satellite, ground_indices[start:stop],
                g_lists, g_top, scores, scale, config, device)
            s_local, s_fusion, s_info = direction_losses(model, satellite, ground, satellite_indices[start:stop],
                s_lists, s_top, scores.T, scale, config, device)
            local_loss, fusion_loss = (g_local + s_local) / 2, (g_fusion + s_fusion) / 2
            total = config["loss_weights"]["local"] * local_loss + config["loss_weights"]["fusion"] * fusion_loss
        rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        previous_scale = scaler.get_scale()
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_value_(model.parameters(), config["clip_grad"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        diagnostics = torch.cat((g_info.pop("diagnostics"), s_info.pop("diagnostics")))
        row = {"epoch": epoch, "step": step, "loss": float(total.detach()),
               "local_loss": float(local_loss.detach()), "fusion_loss": float(fusion_loss.detach()),
               "learning_rates": rates, "grad_scaler": scaler.get_scale(),
               "updated": scaler.get_scale() >= previous_scale, "ground": g_info, "satellite": s_info,
               "transport_max_row_error": float(diagnostics[:, 0].max()),
               "transport_max_column_error": float(diagnostics[:, 1].max()),
               "transport_max_iterations": int(diagnostics[:, 2].max()),
               "transport_unconverged": int((diagnostics[:, 3] == 0).sum())}
        if not torch.isfinite(total.detach()):
            row["loss"] = None
        for key in ("local_loss", "fusion_loss"):
            if not torch.isfinite(torch.tensor(row[key])):
                row[key] = None
        rows.append(row)
        with (output / "local_steps.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    return rows, step


def build_global(config, original, load_initial=True):
    model = TimmModel(original.model, pretrained=load_initial, img_size=original.img_size)
    if load_initial:
        if sha256(config["cvusa_checkpoint"]) != config["cvusa_sha256"]:
            raise ValueError("CVUSA 权重 SHA256 不符合复现记录")
        state = torch.load(config["cvusa_checkpoint"], map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise ValueError(f"原有参数没有从 CVUSA 加载: {missing}")
        print(json.dumps({"cvusa_unexpected_keys": unexpected}), flush=True)
    return model


def run_training(config, output):
    original = reproduction_configuration(config)
    backend = initialize(config["seed"])
    model = build_global(config, original)
    data_config = model.get_config()
    global_model = torch.nn.DataParallel(model, device_ids=list(original.gpu_ids)).to(original.device)
    datasets, manifest, transforms = make_datasets(config, data_config)
    loaders = make_loaders(datasets, config)
    if manifest["splits"]["train_60.csv"]["rows"] != 2472 or manifest["splits"]["val_60.csv"]["rows"] != 1649:
        raise ValueError("训练或测试数量与复现记录不一致")
    criterion = InfoNCE(torch.nn.CrossEntropyLoss(label_smoothing=original.label_smoothing), device=original.device)
    optimizer = torch.optim.AdamW(global_model.parameters(), lr=original.lr)
    schedule_steps, warmup_steps = len(loaders["train"]) * original.epochs, len(loaders["train"]) * original.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, schedule_steps)
    scaler = GradScaler(init_scale=2. ** 10)
    with IsolatedRandomState():
        torch.manual_seed(config["module_init_seed"])
        local_model = LocalMatcher(config).to(original.device)
        local_optimizer = create_local_optimizer(local_model, config)
        local_scheduler = get_cosine_schedule_with_warmup(local_optimizer, warmup_steps, schedule_steps)
        local_scaler = GradScaler(init_scale=config["grad_scaler_init"])
    run = {"scheme": config["scheme"], "config": config, "environment": environment(), "backend": backend,
           "split_manifest": manifest, "transforms": transforms, "scheduler": {
               "warmup_steps": warmup_steps, "schedule_steps": schedule_steps},
           "global_model_structure": str(model), "initial_global_sha256": model_fingerprint(model)}
    source = REPOSITORY / "Author CVDisaster" / "geolocalization"
    run["global_source_sha256"] = {str(path.relative_to(source)): sha256(path) for path in (
        source / "1_train_cvdisaster.py", source / "sample4geo" / "model.py",
        source / "sample4geo" / "trainer.py", source / "sample4geo" / "loss.py")}
    test_g, test_s, test_result, _ = cached_global_evaluation(original, global_model, loaders, "test", "initial", config)
    train_g, train_s, train_result, neighbours = cached_global_evaluation(original, global_model, loaders, "train", "initial", config)
    run["initial_global_test"] = test_result["metrics"]
    run["initial_global_train"] = train_result["metrics"]
    write_json(output / "run.json", run)
    shuffle_training(datasets["train"], neighbours, config)
    del test_g, test_s, test_result, train_g, train_s, train_result
    global_step, local_step, best_global, best_fusion = 0, 0, 0.0, 0.0
    history = []
    for epoch in range(1, original.epochs + 1):
        started = time.perf_counter()
        sample_ids = [int(value) for value in datasets["train"].samples]
        global_rows = []
        previous_scale = [scaler.get_scale()]
        def record_global(loss, ground, satellite):
            value = float(loss.detach())
            scale_value = scaler.get_scale()
            global_rows.append({"loss": value if math.isfinite(value) else None,
                                "lr": optimizer.param_groups[0]["lr"], "grad_scaler": scale_value,
                                "updated": scale_value >= previous_scale[0]})
            previous_scale[0] = scale_value
        loss = author_train(original, global_model, loaders["train"], criterion, optimizer, scheduler, scaler,
                            step_callback=record_global)
        global_step += len(global_rows)
        context = f"epoch{epoch}:global_step{global_step}"
        test_g, test_s, test_result, _ = cached_global_evaluation(original, global_model, loaders, "test", context, config)
        train_g, train_s, train_result, neighbours = cached_global_evaluation(original, global_model, loaders, "train", context, config)
        global_r1 = test_result["metrics"]["R@1"]
        if global_r1 > best_global:
            best_global = global_r1
            save_checkpoint(output / "best_global.pth", global_model, local_model, config, manifest,
                            epoch, global_step, local_step, test_result["metrics"], global_only=True)
        shuffle_training(datasets["train"], neighbours, config)
        before = model_fingerprint(model)
        with IsolatedRandomState():
            local_rows, local_step = train_local_epoch(local_model, (train_g, train_s), train_result, sample_ids,
                model.logit_scale.exp().detach(), config, local_optimizer, local_scheduler, local_scaler,
                epoch, local_step, output)
            result = rerank(local_model, test_g, test_s, test_result["scores"], config, original.device)
        if model_fingerprint(model) != before:
            raise AssertionError("新增模块训练修改了原 geo 参数或 buffer")
        if len(local_rows) != len(global_rows):
            raise AssertionError("全局与新增模块的每轮 batch 数不一致")
        write_json(output / f"queries_epoch_{epoch}.json", {"ids": test_g.ids.tolist(), **result})
        if result["metrics"]["R@1"] > best_fusion:
            best_fusion = result["metrics"]["R@1"]
            save_checkpoint(output / "best_fusion.pth", global_model, local_model, config, manifest,
                            epoch, global_step, local_step, result["metrics"])
            write_json(output / "best_queries.json", {"ids": test_g.ids.tolist(), **result})
        record = {"epoch": epoch, "global_step": global_step, "local_step": local_step,
                  "sample_ids": sample_ids, "global_loss": loss if math.isfinite(loss) else None,
                  "global_steps": global_rows,
                  "global_metrics": test_result["metrics"], "fusion_metrics": result["metrics"],
                  "repaired": result["repaired"], "harmed": result["harmed"],
                  "global_unchanged_during_local_training": True, "seconds": time.perf_counter() - started}
        history.append(record)
        write_json(output / "history.json", history)
        del test_g, test_s, test_result, train_g, train_s, train_result, result, local_rows
    save_checkpoint(output / "last.pth", global_model, local_model, config, manifest,
                    original.epochs, global_step, local_step, history[-1]["fusion_metrics"])
    if best_global <= 0 or best_fusion <= 0:
        raise RuntimeError("训练没有产生有效的全局或融合 best")
    saved = torch.load(output / "best_fusion.pth", map_location="cpu")
    model.load_state_dict(saved["global"], strict=True)
    local_model.load_state_dict(saved["local"], strict=True)
    with IsolatedRandomState():
        g, s, global_result, _ = cached_global_evaluation(original, global_model, loaders, "test", "reload_best_fusion", config)
        result = rerank(local_model, g, s, global_result["scores"], config, original.device)
        expected = json.loads((output / "best_queries.json").read_text(encoding="utf-8"))
        if result["ranks"] != expected["ranks"] or result["metrics"] != expected["metrics"]:
            raise AssertionError("best_fusion 保存加载后排名不一致")
        fixed = {str(beta): fixed_fusion_metrics(result, g, s, global_result["scores"], beta)
                 for beta in (0.3, 0.4)}
    write_json(output / "result.json", {"metrics": result["metrics"], "global_metrics": global_result["metrics"],
        "baseline_metrics": config["baseline_metrics"],
        "difference_percentage_points": {key: value - config["baseline_metrics"][key]
                                         for key, value in result["metrics"].items()},
        "best_global_score": best_global, "best_fusion_epoch": saved["epoch"], "fixed_fusion_metrics": fixed,
        "repaired": result["repaired"], "harmed": result["harmed"], "global_top20_recall": result["global_topk_recall"],
        "checkpoint_reload_verified": True, "best_fusion_sha256": sha256(output / "best_fusion.pth")})


def run_evaluation(checkpoint_path, output):
    saved = torch.load(checkpoint_path, map_location="cpu")
    config = saved["config"]
    original = reproduction_configuration(config)
    initialize(config["seed"])
    model = build_global(config, original, load_initial=False)
    model.load_state_dict(saved["global"], strict=True)
    global_model = torch.nn.DataParallel(model, device_ids=list(original.gpu_ids)).to(original.device)
    datasets, manifest, transforms = make_datasets(config, model.get_config())
    if manifest != saved["split_manifest"]:
        raise ValueError("评价数据清单与 checkpoint 不一致")
    write_json(output / "run.json", {"environment": environment(), "config": config,
        "split_manifest": manifest, "transforms": transforms, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path), "checkpoint_epoch": saved["epoch"]})
    if saved["global_only"]:
        g, _, global_result, _ = cached_global_evaluation(original, global_model, make_loaders(datasets, config),
                                                         "test", sha256(checkpoint_path), config)
        write_json(output / "result.json", {"ids": g.ids.tolist(), "metrics": global_result["metrics"],
                                            "global_only": True, "checkpoint": str(checkpoint_path),
                                            "checkpoint_sha256": sha256(checkpoint_path)})
        return
    with IsolatedRandomState():
        local_model = LocalMatcher(config).to(original.device)
        local_model.load_state_dict(saved["local"], strict=True)
        g, s, global_result, _ = cached_global_evaluation(original, global_model, make_loaders(datasets, config),
                                                        "test", sha256(checkpoint_path), config)
        result = rerank(local_model, g, s, global_result["scores"], config, original.device)
    write_json(output / "result.json", {"ids": g.ids.tolist(), "global_metrics": global_result["metrics"], **result})


def main():
    parser = argparse.ArgumentParser(description="全局训练与局部候选判断独立的 geo 实验")
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "geo_partial_match_gate.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    require_compute_node()
    output = new_output(args.output)
    if args.mode == "train":
        run_training(json.loads(Path(args.config).read_text(encoding="utf-8")), output)
    else:
        if args.checkpoint is None:
            parser.error("评价必须提供 checkpoint")
        run_evaluation(Path(args.checkpoint).resolve(), output)
