import argparse
from pathlib import Path

import torch

from geolocalization.author_validation import validate_author_evaluation
from geolocalization.data import make_datasets, make_loaders
from geolocalization.evaluation import evaluate, save_predictions
from geolocalization.runtime import (create_model, environment, initialize, memory_usage,
                                     new_output, parallel_model, require_compute_node, sha256, write_json)


def main():
    parser = argparse.ArgumentParser(description="重新计算完整测试集指标和 Top-K 预测")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require_compute_node()
    output = new_output(args.output)
    checkpoint_path = Path(args.checkpoint).resolve()
    saved = torch.load(checkpoint_path, map_location="cpu")
    config = saved["config"]
    initialize(saved["seed"])
    model = create_model(config, saved["architecture"])
    model.load_state_dict(saved["model"], strict=True)
    if model.fusion_state() != saved["fusion"]:
        raise ValueError("checkpoint 的融合参数与保存记录不一致")
    datasets, manifest, transforms = make_datasets(config, model.get_config())
    if manifest != saved["metadata"]["split_manifest"]:
        raise ValueError("评价时的数据划分与 checkpoint 的实验记录不一致")
    loaders = make_loaders(datasets, config, saved["seed"])
    model = parallel_model(model)
    result = evaluate(model, loaders, checkpoint_id=sha256(checkpoint_path))
    if saved["architecture"] == "baseline":
        result["author_evaluation_consistency"] = validate_author_evaluation(
            checkpoint_path, loaders, result, config, output)
    summary = save_predictions(result, datasets["test_ground"], output, config)
    summary.update(checkpoint=str(checkpoint_path), checkpoint_epoch=saved["epoch"],
                   seed=saved["seed"], architecture=saved["architecture"], scheme=config["scheme"],
                   environment=environment(), memory=memory_usage(), transforms=transforms)
    write_json(output / "result.json", summary)


if __name__ == "__main__":
    main()
