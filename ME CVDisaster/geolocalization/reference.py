from pathlib import Path

import torch

from .author_validation import validate_author_evaluation
from .evaluation import evaluate, save_predictions
from .runtime import create_model, parallel_model, sha256, write_json


def evaluate_baseline_reference(config, loaders, dataset, output, expected_sha256):
    reference = config["baseline_reference"]
    checkpoint_path = Path(reference["checkpoint"])
    checkpoint_sha256 = sha256(checkpoint_path)
    if checkpoint_sha256 != expected_sha256:
        raise ValueError("已有基线 checkpoint 在本次运行期间发生变化")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = create_model(config, "baseline")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    model = parallel_model(model)
    # 读取已完成实验的 best checkpoint，仅执行预测与指标核验。
    result = evaluate(model, loaders, checkpoint_id=checkpoint_sha256)
    result["author_evaluation_consistency"] = validate_author_evaluation(
        checkpoint_path, loaders, result, config, output, checkpoint_format="author")
    for name, expected in reference["metrics"].items():
        if abs(result["metrics"][name] - expected) > 5e-5:
            write_json(output / "measured_metrics.json", result["metrics"])
            raise ValueError(f"已有基线 checkpoint 的 {name} 与作业 3979539 记录不一致")
    summary = save_predictions(result, dataset, output, config)
    summary.update(architecture="baseline", seed=reference["seed"], scheme=config["scheme"],
                   job_id=reference["job_id"], training_commit=reference["training_commit"],
                   checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_sha256,
                   parameter_updates=0)
    write_json(output / "result.json", summary)
    return summary
