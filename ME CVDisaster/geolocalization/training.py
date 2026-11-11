import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import torch
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

from .comparison import compare_results
from .data import make_datasets, make_loaders, shuffle_training
from .evaluation import evaluate, mine_neighbours, save_predictions
from .loss import contrastive_loss
from .reference import evaluate_baseline_reference
from .runtime import (environment, initialize, load_initial_model, memory_usage, new_output,
                      parallel_model, random_state, read_config, require_compute_node, sha256, write_json)


def check_gradients(model):
    groups = {"model": 0.0, "logit_scale": 0.0}
    if model.architecture == "geo_cvft_refine":
        groups.update(channel_projection=0.0, feature_transport=0.0, refinement=0.0,
                      fusion_logit=0.0)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"参数梯度缺失或包含非有限数值: {name}")
        group = name.partition(".")[0]
        groups[group] = max(groups[group], float(parameter.grad.detach().abs().max()))
    if any(value == 0 for value in groups.values()):
        raise FloatingPointError(f"功能模块梯度为零: {groups}")
    return groups


def train_epoch(model, loader, optimizer, scheduler, scaler, config, epoch, output, step):
    model.train()
    rows = []
    seen_ids = []
    diagnostic_values = []
    started = time.perf_counter()
    for ground_image, satellite_image, sample_ids in loader:
        if len(sample_ids) != config["batch_size"]:
            raise ValueError("每个训练 batch 必须包含 64 对图片")
        seen_ids.extend(sample_ids.tolist())
        sample_ids = sample_ids.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=config["mixed_precision"]):
            ground, satellite = model(ground_image.cuda(non_blocking=True),
                                      satellite_image.cuda(non_blocking=True), sample_ids=sample_ids)
        # DataParallel 已经汇集两张卡的特征，在主卡形成完整的 64×64 logits。
        loss = contrastive_loss(ground, satellite, model.module, sample_ids, config["label_smoothing"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = check_gradients(model.module)
        fusion_gradient = (float(model.module.fusion_logit.grad.detach())
                           if model.module.architecture == "geo_cvft_refine" else None)
        fusion_before = model.module.fusion_state()
        torch.nn.utils.clip_grad_value_(model.parameters(), config["clip_grad"])
        learning_rate = optimizer.param_groups[0]["lr"]
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        row = {"epoch": epoch, "step": step, "loss": float(loss.detach()),
               "lr": learning_rate, "next_lr": optimizer.param_groups[0]["lr"],
               "logit_scale": float(model.module.logit_scale.detach()),
               "grad_scaler": scaler.get_scale(), "gradients": gradients}
        if model.module.architecture == "geo_cvft_refine":
            row.update(fusion_before=fusion_before, fusion_after=model.module.fusion_state(),
                       fusion_logit_gradient=fusion_gradient)
            values = ground["transport_diagnostics"].detach().cpu()
            diagnostic_values.append(values)
            row["transport"] = {"mean_row_entropy": float(values[:, 0].mean()),
                                "max_element": float(values[:, 1].max()),
                                "e_row": float(values[:, 2].max()), "e_col": float(values[:, 3].max())}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
        with (output / "steps.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    if seen_ids != loader.dataset.samples or len(rows) != config["steps_per_epoch"]:
        raise ValueError("训练执行顺序或优化步数与本轮采样记录不匹配")
    report = {"epoch": epoch, "step": step, "steps": len(rows),
              "mean_loss": sum(row["loss"] for row in rows) / len(rows),
              "seconds": time.perf_counter() - started, "sample_ids": seen_ids}
    report["fusion"] = model.module.fusion_state()
    if diagnostic_values:
        values = torch.cat(diagnostic_values)
        report["transport"] = {"mean_row_entropy": float(values[:, 0].mean()),
                               "mean_max_element": float(values[:, 1].mean()),
                               "max_element": float(values[:, 1].max()),
                               "e_row": float(values[:, 2].max()), "e_col": float(values[:, 3].max())}
    return report, step


def checkpoint(model, optimizer, scheduler, scaler, epoch, step, best_epoch, best_score,
               config, architecture, seed, loaders, metadata, samples):
    return {"model": model.module.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "step": step, "best_epoch": best_epoch, "best_score": best_score,
            "config": config, "architecture": architecture, "seed": seed,
            "random_state": random_state(loaders), "metadata": metadata, "sample_ids": samples,
            "fusion": model.module.fusion_state()}


def run_training(args):
    require_compute_node()
    config = read_config(args.config)
    if args.seed != config["seed"] or args.architecture != config["training_architecture"]:
        raise ValueError("本次正式训练要求 geo_cvft_refine、seed=42")
    baseline_checkpoint_sha256 = sha256(config["baseline_reference"]["checkpoint"])
    output = new_output(args.output)
    backend = initialize(args.seed)
    started = time.perf_counter()
    model, initial_scales = load_initial_model(config, args.architecture)
    data_config = model.get_config()
    datasets, manifest, transforms = make_datasets(config, data_config)
    loaders = make_loaders(datasets, config, args.seed)
    model = parallel_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]),
                                  eps=config["eps"], weight_decay=config["weight_decay"],
                                  amsgrad=config["amsgrad"])
    # 调度长度在首次自定义采样之前计算，实际采样每轮产生 38 次更新。
    scheduled_steps = len(loaders["train"]) * config["epochs"]
    warmup_steps = len(loaders["train"])
    if (scheduled_steps, warmup_steps) != (config["scheduler_steps"], config["warmup_steps"]):
        raise ValueError("调度步数与方案不符")
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, scheduled_steps)
    scaler = GradScaler(init_scale=config["grad_scaler_init"], enabled=config["mixed_precision"])
    metadata = {"environment": environment(), "backend": backend, "data_config": data_config,
                "initial_logit_scales": initial_scales, "split_manifest": manifest,
                "initial_fusion": model.module.fusion_state(),
                "baseline_reference": config["baseline_reference"],
                "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "transforms": transforms, "architecture": args.architecture, "seed": args.seed,
                "scheme": config["scheme"], "config": config}
    write_json(output / "run.json", metadata)
    for name in manifest["splits"]:
        shutil.copyfile(Path(config["data_root"]) / name, output / name)
    slurm_script = metadata["environment"]["slurm_script"]
    if slurm_script:
        shutil.copyfile(slurm_script, output / "submitted.sbatch")
    run_id = f"{metadata['environment']['commit']}:{args.architecture}:seed{args.seed}"
    zero_shot = evaluate(model, loaders, checkpoint_id=f"{run_id}:initial")
    write_json(output / "zero_shot.json", {"metrics": zero_shot["metrics"], "transport": zero_shot["transport"],
                                          "feature_context": zero_shot["feature_context"]})
    del zero_shot
    neighbours = mine_neighbours(model, loaders, config)
    best_score, best_epoch, best_ranks = -1.0, None, None
    history = []
    step = 0
    for epoch in range(1, config["epochs"] + 1):
        sample_ids = shuffle_training(datasets["train"], neighbours, config)
        write_json(output / f"samples_epoch_{epoch:02d}.json", sample_ids)
        report, step = train_epoch(model, loaders["train"], optimizer, scheduler, scaler,
                                    config, epoch, output, step)
        result = evaluate(model, loaders, checkpoint_id=f"{run_id}:epoch{epoch}:step{step}")
        report["evaluation"] = result["metrics"]
        report["evaluation_timing"] = result["timing"]
        report["feature_context"] = result["feature_context"]
        report["memory"] = memory_usage()
        history.append(report)
        write_json(output / "history.json", history)
        if result["metrics"]["R@1"] > best_score:
            best_score, best_epoch = result["metrics"]["R@1"], epoch
            best_ranks = result["ranks"].clone()
            torch.save(checkpoint(model, optimizer, scheduler, scaler, epoch, step, best_epoch,
                                  best_score, config, args.architecture, args.seed, loaders, metadata,
                                  sample_ids), output / "best.pth")
        del result
        if epoch < config["epochs"]:
            neighbours = mine_neighbours(model, loaders, config)
    if step != config["steps_per_epoch"] * config["epochs"]:
        raise ValueError(f"训练总优化步数错误: {step}")
    torch.save(checkpoint(model, optimizer, scheduler, scaler, epoch, step, best_epoch,
                          best_score, config, args.architecture, args.seed, loaders, metadata,
                          sample_ids), output / "last.pth")
    best = torch.load(output / "best.pth", map_location="cpu")
    model.module.load_state_dict(best["model"], strict=True)
    if model.module.fusion_state() != best["fusion"]:
        raise RuntimeError("best checkpoint 中的融合参数与记录不一致")
    result = evaluate(model, loaders, checkpoint_id=sha256(output / "best.pth"))
    if result["metrics"]["R@1"] != best_score or not torch.equal(result["ranks"], best_ranks):
        raise RuntimeError("重新加载 best 后的完整查询排名与保存时不一致")
    summary = save_predictions(result, datasets["test_ground"], output, config)
    summary.update(best_epoch=best_epoch, total_steps=step, architecture=args.architecture,
                   seed=args.seed, scheme=config["scheme"],
                   training_seconds=time.perf_counter() - started, memory=memory_usage(),
                   best_checkpoint=str(output / "best.pth"), last_checkpoint=str(output / "last.pth"),
                   checkpoint_reload_verified=True)
    write_json(output / "result.json", summary)
    # 改进模型训练和 best 评价完成后，读取已有基线权重补充逐查询预测。
    # 基线仅参与评价；模型、optimizer 和 GradScaler 的训练资源在此释放。
    del result, best, model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    baseline_summary = evaluate_baseline_reference(
        config, loaders, datasets["test_ground"], output / "baseline_reference", baseline_checkpoint_sha256)
    comparison = compare_results(output, output / "baseline_reference/best_queries.csv", output / "comparison")
    summary.update(baseline_reference_result=str(output / "baseline_reference/result.json"),
                   baseline_author_evaluation_verified=baseline_summary["author_evaluation_consistency"]["passed"],
                   comparison=str(output / "comparison/comparison.json"),
                   metric_differences={name: values["difference_percentage_points"]
                                       for name, values in comparison["metrics"].items()},
                   total_run_seconds=time.perf_counter() - started)
    write_json(output / "result.json", summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="训练 geo_cvft_refine，使用 seed=42 与已有基线比较")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/geo_cvft_refine.json"))
    parser.add_argument("--architecture", choices=("geo_cvft_refine",), default="geo_cvft_refine")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--output", required=True)
    run_training(parser.parse_args())
