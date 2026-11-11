import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast

from sample4geo.evaluate.cvusa_and_cvact import calculate_nearest
from .model import normalize_features
from .runtime import write_json


@torch.no_grad()
def predict(model, dataloader, view):
    model.eval()
    collected = {}
    labels = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for image, sample_ids in dataloader:
        with autocast():
            result = model(image.cuda(non_blocking=True), view=view,
                           sample_ids=sample_ids.cuda(non_blocking=True))
        with autocast(enabled=False):
            result["descriptors"] = normalize_features(result["descriptors"], "evaluation_descriptors")
        for name, tensor in result.items():
            # 评价仅保留向量和诊断量，搬移矩阵随当前 batch 释放。
            if name != "transport":
                collected.setdefault(name, []).append(tensor.detach().float().cpu())
        labels.append(sample_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {name: torch.cat(values) for name, values in collected.items()}
    ids = torch.cat(labels)
    if len(ids.unique()) != len(ids):
        raise ValueError("评价数据包含重复图片 ID")
    for name in ("global_features", "spatial_features", "descriptors"):
        if name in result:
            tensor = result[name]
            if not torch.isfinite(tensor).all() or not torch.allclose(
                    tensor.norm(dim=1), torch.ones(len(ids)), atol=1e-5, rtol=1e-5):
                raise FloatingPointError(f"{view} {name} 数值或范数错误")
    return result, ids, elapsed


def retrieval_scores(ground, satellite, ground_ids, satellite_ids):
    with autocast(enabled=False):
        # predict 已经统一完成 FP32 归一化，此处使用同一批候选缓存向量。
        ground = ground.float()
        satellite = satellite.float()
        similarities = ground @ satellite.T
    metrics, ranks = rank_similarities(similarities, ground_ids, satellite_ids)
    return metrics, ranks, similarities


def rank_similarities(similarities, ground_ids, satellite_ids):
    if len(ground_ids) != len(satellite_ids) or set(ground_ids.tolist()) != set(satellite_ids.tolist()):
        raise ValueError("查询与候选的图片 ID 集合不匹配")
    if len(ground_ids.unique()) != len(ground_ids) or len(satellite_ids.unique()) != len(satellite_ids):
        raise ValueError("评价要求每个图片 ID 唯一")
    reference_index = {int(value): index for index, value in enumerate(satellite_ids)}
    matches = torch.tensor([reference_index[int(value)] for value in ground_ids], device=similarities.device)
    if similarities.shape != (len(ground_ids), len(satellite_ids)) or not torch.isfinite(similarities).all():
        raise FloatingPointError("检索相似度矩阵的形状或数值错误")
    correct_scores = similarities[torch.arange(len(matches), device=similarities.device), matches]
    # 保留作者定义：只计算严格大于正确图片分数的候选数量。
    ranks = (similarities > correct_scores[:, None]).sum(1)
    cutoffs = {"R@1": 1, "R@5": 5, "R@10": 10, "R@1%": len(satellite_ids) // 100}
    metrics = {name: float((ranks < cutoff).double().mean() * 100)
               for name, cutoff in cutoffs.items()}
    return metrics, ranks


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


def save_candidate_cache(path, features, ids, context):
    verify_feature_context(features, context)
    torch.save({"features": features, "ids": ids, "feature_context": context}, path)


def load_candidate_cache(path, expected_context):
    saved = torch.load(path, map_location="cpu")
    if saved["feature_context"] != expected_context:
        raise ValueError("候选缓存的 checkpoint 或融合比例与当前模型不一致，需要重新计算候选特征")
    verify_feature_context(saved["features"], expected_context)
    if len(saved["ids"]) != len(saved["features"]["descriptors"]) or len(saved["ids"].unique()) != len(saved["ids"]):
        raise ValueError("候选缓存的图片 ID 数量或唯一性错误")
    return saved


def mine_neighbours(model, loaders, config):
    satellite, satellite_ids, _ = predict(model, loaders["train_satellite"], "satellite")
    ground, ground_ids, _ = predict(model, loaders["train_ground"], "ground")
    # 调用原有 topk 实现，保持作者难负候选的顺序规则。
    return calculate_nearest(ground["descriptors"], satellite["descriptors"], ground_ids,
                             satellite_ids, neighbour_range=config["neighbour_range"])


def save_predictions(result, dataset, output, config, prefix="best"):
    output = Path(output)
    summary = {name: result[name] for name in
               ("metrics", "transport", "timing", "descriptor_dimension", "feature_context")}
    fusion_weight = result["feature_context"]["fusion_weight"]
    summary["fusion"] = {name: result["feature_context"][name] for name in ("fusion_logit", "fusion_weight")}
    if "author_evaluation_consistency" in result:
        summary["author_evaluation_consistency"] = result["author_evaluation_consistency"]
    cache_path = output / f"{prefix}_satellite_features.pt"
    save_candidate_cache(cache_path, result["satellite"], result["satellite_ids"], result["feature_context"])
    loaded_cache = load_candidate_cache(cache_path, result["feature_context"])
    torch.testing.assert_close(loaded_cache["features"]["descriptors"], result["satellite"]["descriptors"], rtol=0, atol=0)
    if not torch.equal(loaded_cache["ids"], result["satellite_ids"]):
        raise RuntimeError("候选缓存保存加载后的 ID 顺序不一致")
    summary["candidate_cache"] = str(cache_path)
    summary["candidate_cache_verified"] = True
    for branch in ("global_features", "spatial_features"):
        if branch in result["ground"]:
            metrics, _, _ = retrieval_scores(result["ground"][branch], result["satellite"][branch],
                                             result["ground_ids"], result["satellite_ids"])
            summary[branch] = metrics
    frame = gpd.read_file(Path(config["data_root"]) / "02_Position" / "CVIAN_position.geojson")
    if frame.crs is None:
        raise ValueError("位置数据缺少 CRS")
    frame = frame.to_crs(epsg=4326)
    frame["image_id"] = frame["id"].map(int)
    if frame["image_id"].duplicated().any():
        raise ValueError("位置数据包含重复图片 ID")
    positions = {int(row.image_id): (float(row.geometry.x), float(row.geometry.y))
                 for row in frame.itertuples()}
    reference_ids = result["satellite_ids"].numpy()
    # 同分候选采用数值 ID 升序；作者指标的严格比较规则独立保存。
    id_order = np.argsort(reference_ids, kind="stable")
    query_rows, candidate_rows = [], []
    branch_scores = {name: result["ground"][name] @ result["satellite"][name].T
                     for name in ("global_features", "spatial_features") if name in result["ground"]}
    for index, query_id in enumerate(result["ground_ids"].tolist()):
        scores = result["scores"][index].numpy()
        order = id_order[np.argsort(-scores[id_order], kind="stable")]
        correct_index = int(np.flatnonzero(reference_ids == query_id)[0])
        query_rows.append({"query_id": query_id, "correct_rank": int(result["ranks"][index]),
                           "correct_score": float(scores[correct_index]),
                           "correct_score_ties": int(np.count_nonzero(scores == scores[correct_index])),
                           "top1_id": int(reference_ids[order[0]]), "fusion_weight": fusion_weight,
                           "ground_path": str(Path(config["data_root"]) / dataset.idx2ground[query_id]),
                           "correct_satellite_path": str(Path(config["data_root"]) / dataset.idx2sat[query_id])})
        for rank, candidate_index in enumerate(order[:config["top_k"]], start=1):
            candidate_id = int(reference_ids[candidate_index])
            longitude, latitude = positions[candidate_id]
            if not np.isfinite([longitude, latitude]).all():
                raise ValueError(f"图片 {candidate_id} 的经纬度包含非有限数值")
            row = {"query_id": query_id, "rank": rank, "satellite_id": candidate_id,
                   "longitude": longitude, "latitude": latitude, "score": float(scores[candidate_index]),
                   "fusion_weight": fusion_weight,
                   "score_ties": int(np.count_nonzero(scores == scores[candidate_index])),
                   "ground_path": str(Path(config["data_root"]) / dataset.idx2ground[query_id]),
                   "satellite_path": str(Path(config["data_root"]) / dataset.idx2sat[candidate_id])}
            for branch, values in branch_scores.items():
                row[branch + "_score"] = float(values[index, candidate_index])
            candidate_rows.append(row)
    pd.DataFrame(query_rows).to_csv(output / f"{prefix}_queries.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output / f"{prefix}_topk.csv", index=False)
    summary["rank_definition"] = "number_of_candidates_with_strictly_greater_score"
    summary["topk_tie_order"] = "satellite_id_numeric_ascending"
    write_json(output / f"{prefix}_metrics.json", summary)
    return summary
