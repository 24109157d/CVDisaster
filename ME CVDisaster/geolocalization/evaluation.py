import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from sample4geo.evaluate.cvusa_and_cvact import calculate_nearest, calculate_scores
from sample4geo.trainer import predict as author_predict


@torch.no_grad()
def predict(model, dataloader, view, keep_on_gpu=False):
    model.eval()
    collected = {}

    def extract_features(current_model, image, sample_ids):
        result = current_model(image, view=view, sample_ids=sample_ids.to(image.device))
        for name, tensor in result.items():
            if name not in ("transport", "descriptors"):
                collected.setdefault(name, []).append(tensor.detach().float().cpu())
        return result["descriptors"]

    torch.cuda.synchronize()
    started = time.perf_counter()
    prediction_config = SimpleNamespace(device="cuda", verbose=True, normalize_features=True)
    descriptors, ids = author_predict(prediction_config, model, dataloader, feature_extractor=extract_features)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {name: torch.cat(values) for name, values in collected.items()}
    result["descriptors"] = descriptors if keep_on_gpu else descriptors.cpu()
    ids = ids if keep_on_gpu else ids.cpu()
    return result, ids, elapsed


def retrieval_scores(ground, satellite, ground_ids, satellite_ids):
    result = calculate_scores(ground.to("cuda", dtype=torch.float32), satellite.to("cuda", dtype=torch.float32),
                              ground_ids.to("cuda"), satellite_ids.to("cuda"),
                              step_size=1000, ranks=[1, 5, 10], return_details=True)
    return result["metrics"], result["ranks"], result["scores"]


def feature_context(model, checkpoint_id):
    if not checkpoint_id:
        raise ValueError("特征计算必须指定 checkpoint 标识")
    return {"checkpoint_id": checkpoint_id, "architecture": model.module.architecture,
            **model.module.fusion_state()}


def verify_feature_context(features, context):
    if context["architecture"] == "geo_cvft_refine":
        weights = features["fusion_weight"]
        if not torch.isfinite(weights).all() or not (weights == context["fusion_weight"]).all():
            raise RuntimeError("候选或查询的融合比例与 checkpoint 不一致")


def evaluate(model, loaders, *, checkpoint_id, split="test"):
    context = feature_context(model, checkpoint_id)
    satellite, satellite_ids, satellite_seconds = predict(model, loaders[split + "_satellite"], "satellite")
    ground, ground_ids, ground_seconds = predict(model, loaders[split + "_ground"], "ground")
    if feature_context(model, checkpoint_id) != context:
        raise RuntimeError("查询与候选特征计算期间融合参数发生变化")
    verify_feature_context(satellite, context)
    verify_feature_context(ground, context)
    metrics, ranks, scores = retrieval_scores(ground["descriptors"], satellite["descriptors"],
                                            ground_ids, satellite_ids)
    diagnostics = {}
    if "transport_diagnostics" in ground:
        values = ground["transport_diagnostics"]
        diagnostics = {"mean_row_entropy": float(values[:, 0].mean()),
                       "mean_max_element": float(values[:, 1].mean()),
                       "max_element": float(values[:, 1].max()),
                       "e_row": float(values[:, 2].max()), "e_col": float(values[:, 3].max())}
    return {"metrics": metrics, "ranks": ranks, "scores": scores,
            "ground": ground, "satellite": satellite, "ground_ids": ground_ids,
            "satellite_ids": satellite_ids, "transport": diagnostics,
            "timing": {"ground_seconds": ground_seconds, "satellite_seconds": satellite_seconds,
                       "ground_seconds_per_image": ground_seconds / len(ground_ids),
                       "satellite_seconds_per_image": satellite_seconds / len(satellite_ids)},
            "descriptor_dimension": ground["descriptors"].shape[1], "feature_context": context}


def mine_neighbours(model, loaders, config):
    satellite, satellite_ids, _ = predict(model, loaders["train_satellite"], "satellite", keep_on_gpu=True)
    ground, ground_ids, _ = predict(model, loaders["train_ground"], "ground", keep_on_gpu=True)
    metrics = calculate_scores(ground["descriptors"], satellite["descriptors"], ground_ids,
                               satellite_ids, step_size=1000, ranks=[1, 5, 10], return_details=True)["metrics"]
    neighbours = calculate_nearest(ground["descriptors"], satellite["descriptors"], ground_ids,
                                   satellite_ids, neighbour_range=config["neighbour_range"])
    return metrics, neighbours


def save_predictions(result, dataset, output, config, prefix="best"):
    output = Path(output)
    summary = {name: result[name] for name in
               ("metrics", "transport", "timing", "descriptor_dimension", "feature_context")}
    summary["fusion"] = {name: result["feature_context"][name] for name in ("fusion_logit", "fusion_weight")}
    for branch in ("global_features", "spatial_features"):
        if branch in result["ground"]:
            metrics, _, _ = retrieval_scores(result["ground"][branch], result["satellite"][branch],
                                             result["ground_ids"], result["satellite_ids"])
            summary[branch] = metrics
    reference_ids = result["satellite_ids"].numpy()
    # 同分候选采用数值 ID 升序；作者指标的严格比较规则独立保存。
    id_order = np.argsort(reference_ids, kind="stable")
    query_rows = []
    branch_scores = {name: result["ground"][name] @ result["satellite"][name].T
                     for name in ("global_features", "spatial_features") if name in result["ground"]}
    for index, query_id in enumerate(result["ground_ids"].tolist()):
        scores = result["scores"][index].numpy()
        order = id_order[np.argsort(-scores[id_order], kind="stable")]
        correct_index = int(np.flatnonzero(reference_ids == query_id)[0])
        selected = order[:config["top_k"]]
        top1_id = int(reference_ids[order[0]])
        row = {"query_id": query_id, "correct_rank": int(result["ranks"][index]),
               "correct_score": float(scores[correct_index]),
               "correct_score_ties": int(np.count_nonzero(scores == scores[correct_index])),
               "top1_id": top1_id, "top1_score": float(scores[order[0]]),
               "topk_ids": json.dumps(reference_ids[selected].tolist()),
               "topk_scores": json.dumps(scores[selected].tolist(), allow_nan=False),
               "ground_path": str(Path(config["data_root"]) / dataset.idx2ground[query_id]),
               "correct_satellite_path": str(Path(config["data_root"]) / dataset.idx2sat[query_id]),
               "top1_satellite_path": str(Path(config["data_root"]) / dataset.idx2sat[top1_id])}
        for branch, values in branch_scores.items():
            row[branch + "_correct_score"] = float(values[index, correct_index])
            row[branch + "_top1_score"] = float(values[index, order[0]])
        if "transport_diagnostics" in result["ground"]:
            for column, name in enumerate(("row_entropy", "max_element", "e_row", "e_col")):
                row[name] = float(result["ground"]["transport_diagnostics"][index, column])
        query_rows.append(row)
    pd.DataFrame(query_rows).to_csv(output / f"{prefix}_queries.csv", index=False)
    summary["rank_definition"] = "number_of_candidates_with_strictly_greater_score"
    summary["topk_tie_order"] = "satellite_id_numeric_ascending"
    return summary
