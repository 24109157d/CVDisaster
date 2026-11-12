import argparse
import gc
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from torch.cuda.amp import GradScaler
from transformers import get_cosine_schedule_with_warmup

from sample4geo.trainer import train as author_train

from .comparison import compare_results
from .data import make_datasets, make_loaders, shuffle_training
from .diagnostics import gradient_statistics
from .evaluation import evaluate, mine_neighbours, save_predictions
from .loss import DescriptorInfoNCE
from .reference import evaluate_baseline_reference
from .runtime import (environment, initialize, load_initial_model, memory_usage, new_output,
                      parallel_model, read_config, require_compute_node, sha256, write_json)


def train_epoch(model, loader, optimizer, scheduler, scaler, config, epoch, output, step):
    rows = []
    seen_ids = []
    diagnostic_values = []
    started = time.perf_counter()
    batch = {}

    def forward_adapter(current_model, ground_image, satellite_image, sample_ids):
        seen_ids.extend(sample_ids.tolist())
        batch["fusion_before"] = current_model.module.fusion_state()
        return current_model(ground_image, satellite_image, sample_ids=sample_ids.to(ground_image.device))

    def record_step(loss, ground, satellite):
        nonlocal step
        gradients = gradient_statistics(model.module)
        fusion_gradient = None
        if model.module.architecture == "geo_cvft_refine":
            value = model.module.fusion_logit.grad
            if value is not None and torch.isfinite(value).all():
                fusion_gradient = float(value.detach())
        learning_rate = optimizer.param_groups[0]["lr"]
        step += 1
        loss_value = float(loss.detach())
        row = {"epoch": epoch, "step": step, "loss": loss_value if math.isfinite(loss_value) else None,
               "lr": learning_rate,
               "logit_scale": float(model.module.logit_scale.detach()),
               "grad_scaler": scaler.get_scale() if scaler is not None else None, "gradients": gradients}
        if model.module.architecture == "geo_cvft_refine":
            row.update(fusion_before=batch["fusion_before"], fusion_after=model.module.fusion_state(),
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

    train_config = SimpleNamespace(device="cuda", verbose=True, gpu_ids=(0, 1),
                                   clip_grad=config["clip_grad"], scheduler="cosine")
    criterion = DescriptorInfoNCE(torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"]),
                                  device="cuda")
    mean_loss = author_train(train_config, model, loader, criterion, optimizer, scheduler, scaler,
                             forward_adapter=forward_adapter, step_callback=record_step)
    report = {"epoch": epoch, "step": step, "steps": len(rows),
              "mean_loss": mean_loss if math.isfinite(mean_loss) else None,
              "seconds": time.perf_counter() - started, "sample_ids": seen_ids}
    report["fusion"] = model.module.fusion_state()
    if diagnostic_values:
        values = torch.cat(diagnostic_values)
        report["transport"] = {"mean_row_entropy": float(values[:, 0].mean()),
                               "mean_max_element": float(values[:, 1].mean()),
                               "max_element": float(values[:, 1].max()),
                               "e_row": float(values[:, 2].max()), "e_col": float(values[:, 3].max())}
    return report, step


def checkpoint(model, epoch, step, best_epoch, best_score, config, architecture, seed, metadata):
    return {"model": model.module.state_dict(),
            "epoch": epoch, "step": step, "best_epoch": best_epoch, "best_score": best_score,
            "config": config, "architecture": architecture, "seed": seed,
            "metadata": {"split_manifest": metadata["split_manifest"]},
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
    loaders = make_loaders(datasets, config)
    model = parallel_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]),
                                  eps=config["eps"], weight_decay=config["weight_decay"],
                                  amsgrad=config["amsgrad"])
    # 沿用原复现，在首次自定义采样之前计算调度长度。
    scheduled_steps = len(loaders["train"]) * config["epochs"]
    warmup_steps = len(loaders["train"]) * config["warmup_epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, scheduled_steps)
    scaler = GradScaler(init_scale=config["grad_scaler_init"]) if config["mixed_precision"] else None
    metadata = {"environment": environment(), "backend": backend, "data_config": data_config,
                "initial_logit_scales": initial_scales, "split_manifest": manifest,
                "initial_fusion": model.module.fusion_state(),
                "baseline_reference": config["baseline_reference"],
                "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "transforms": transforms, "architecture": args.architecture, "seed": args.seed,
                "scheme": config["scheme"], "config": config,
                "scheduler": {"warmup_steps": warmup_steps, "training_steps": scheduled_steps}}
    write_json(output / "run.json", metadata)
    run_id = f"{metadata['environment']['commit']}:{args.architecture}:seed{args.seed}"
    zero_shot = evaluate(model, loaders, checkpoint_id=f"{run_id}:initial")
    metadata["initial_evaluation"] = {"metrics": zero_shot["metrics"], "transport": zero_shot["transport"]}
    del zero_shot
    train_metrics, neighbours = mine_neighbours(model, loaders, config)
    metadata["initial_training_evaluation"] = train_metrics
    write_json(output / "run.json", metadata)
    shuffle_training(datasets["train"], neighbours, config)
    best_score, best_epoch, best_ranks = 0.0, None, None
    history = []
    step = 0
    for epoch in range(1, config["epochs"] + 1):
        report, step = train_epoch(model, loaders["train"], optimizer, scheduler, scaler,
                                    config, epoch, output, step)
        result = evaluate(model, loaders, checkpoint_id=f"{run_id}:epoch{epoch}:step{step}")
        train_metrics, neighbours = mine_neighbours(model, loaders, config)
        report["evaluation"] = result["metrics"]
        report["training_evaluation"] = train_metrics
        report["evaluation_timing"] = result["timing"]
        report["feature_context"] = result["feature_context"]
        report["memory"] = memory_usage()
        history.append(report)
        write_json(output / "history.json", history)
        if result["metrics"]["R@1"] > best_score:
            best_score, best_epoch = result["metrics"]["R@1"], epoch
            best_ranks = result["ranks"].clone()
            torch.save(checkpoint(model, epoch, step, best_epoch, best_score, config,
                                  args.architecture, args.seed, metadata), output / "best.pth")
        del result
        shuffle_training(datasets["train"], neighbours, config)
    torch.save(checkpoint(model, epoch, step, best_epoch, best_score, config,
                          args.architecture, args.seed, metadata), output / "last.pth")
    if best_epoch is None:
        raise RuntimeError("训练未产生 R@1 大于 0 的最佳 checkpoint，请检查训练日志")
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
    evaluate_baseline_reference(
        config, loaders, datasets["test_ground"], output / "baseline_reference", baseline_checkpoint_sha256)
    comparison = compare_results(output, output / "baseline_reference/best_queries.csv")
    summary.update(baseline_reference_result=str(output / "baseline_reference/result.json"),
                   comparison=comparison,
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
