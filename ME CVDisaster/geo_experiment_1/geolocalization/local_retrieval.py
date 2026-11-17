import math

import torch
from torch.nn import functional as F

from .local_cache import ranked_candidates


def score_candidates(model, query, reference, query_indices, candidates, device):
    if query.context != reference.context or query.view == reference.view:
        raise ValueError("候选特征来自不同 checkpoint 或相同视角")
    query_indices = torch.as_tensor(query_indices, dtype=torch.long)
    candidates = torch.as_tensor(candidates, dtype=torch.long)
    if candidates.ndim != 2 or candidates.shape[0] != len(query_indices):
        raise ValueError("查询与候选列表数量不一致")
    unique, inverse = torch.unique(candidates.reshape(-1), sorted=True, return_inverse=True)
    q = model.encode(*query.select(query_indices, device), query.view)
    r = model.encode(*reference.select(unique, device), reference.view)
    count = candidates.shape[1]
    expanded_q = q[:, None].expand(-1, count, -1, -1).reshape(-1, q.shape[1], q.shape[2])
    expanded_r = r[inverse]
    ground, satellite = (expanded_q, expanded_r) if query.view == "ground" else (expanded_r, expanded_q)
    scores, statistics, diagnostics = model.score_pairs(ground, satellite)
    return scores.reshape(len(q), count), statistics.reshape(len(q), count, 6), diagnostics


@torch.no_grad()
def mine_local_candidates(query, reference, indices, candidates, local_scores, config, rng):
    indices = torch.as_tensor(indices, dtype=torch.long)
    if candidates.shape != local_scores.shape or len(candidates) != len(indices):
        raise ValueError("当前 batch 的候选与局部分数形状错误")
    local_order = local_scores.detach().float().cpu().argsort(dim=1, descending=True, stable=True)
    result = []
    for row, query_id in enumerate(query.ids[indices].tolist()):
        correct = reference.lookup[query_id]
        selected = [correct]
        global_order = candidates[row].tolist()
        if config["global_hard_count"]:
            for candidate in global_order:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) == 1 + config["global_hard_count"]:
                    break
        target = len(selected) + config["local_hard_count"]
        if config["local_hard_count"]:
            for position in local_order[row].tolist():
                candidate = global_order[position]
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) == target:
                    break
        if len(selected) != target:
            raise ValueError("主要候选范围不足以选取指定的困难候选")
        excluded = set(global_order + selected)
        pool = [value for value in range(len(reference.ids)) if value not in excluded]
        selected.extend(rng.sample(pool, config["random_negative_count"]))
        if len(set(selected)) != len(selected):
            raise AssertionError("局部训练候选存在重复")
        result.append(selected)
    return torch.tensor(result, dtype=torch.long)


def direction_losses(model, query, reference, indices, fusion_candidates, global_scores, config, device, rng):
    temperatures = (config["local_loss_temperature"], config["fusion_loss_temperature"])
    if any(not math.isfinite(value) or value <= 0 for value in temperatures):
        raise ValueError("新增损失温度必须为有限正数")
    selected = torch.as_tensor(indices, dtype=torch.long)
    candidates = fusion_candidates[selected]
    scores, _, diagnostics = score_candidates(model, query, reference, selected, candidates, device)
    local_candidates = mine_local_candidates(query, reference, selected, candidates, scores, config, rng)
    positions = local_candidates[:, :, None] == candidates[:, None, :]
    contained = positions.any(2)
    local_scores = scores.gather(1, positions.long().argmax(2).to(device))
    extra_rows, extra_columns = (~contained).nonzero(as_tuple=True)
    if len(extra_rows):
        extra_candidates = local_candidates[extra_rows, extra_columns, None]
        extra_scores, _, extra_diagnostics = score_candidates(
            model, query, reference, selected[extra_rows], extra_candidates, device)
        local_scores[extra_rows.to(device), extra_columns.to(device)] = extra_scores.flatten()
        diagnostics = torch.cat((diagnostics, extra_diagnostics))
    labels = torch.zeros(len(selected), dtype=torch.long, device=device)
    local_loss = F.cross_entropy(local_scores / temperatures[0], labels, label_smoothing=config["label_smoothing"])
    correct_ids = query.ids[selected]
    reference_ids = reference.ids[candidates]
    matches = reference_ids == correct_ids[:, None]
    valid = matches.any(1)
    if valid.any():
        valid_indices = selected[valid]
        g_scores = global_scores[valid_indices[:, None], candidates[valid]].to(device)
        fused, beta = model.fuse(g_scores, scores[valid.to(device)])
        targets = matches[valid].long().argmax(1).to(device)
        fusion_loss = F.cross_entropy(fused / temperatures[1], targets, label_smoothing=config["label_smoothing"])
        beta_mean = float(beta.detach().mean())
    else:
        fusion_loss = local_scores.sum() * 0
        beta_mean = None
    return local_loss, fusion_loss, {"valid_fusion_queries": int(valid.sum()), "queries": len(selected),
                                      "beta_mean": beta_mean, "diagnostics": diagnostics.detach().cpu(),
                                      "candidate_ids": reference.ids[local_candidates].tolist()}


def recall_from_ranks(ranks, gallery_size):
    return {**{f"R@{k}": float((ranks < k).double().mean() * 100) for k in (1, 5, 10)},
            "R@1%": float((ranks < gallery_size // 100).double().mean() * 100)}


@torch.no_grad()
def rerank(model, query, reference, global_scores, config, device, fixed_beta=None):
    if fixed_beta is not None and not 0 <= fixed_beta <= 1:
        raise ValueError("固定局部比例超出范围")
    model.eval()
    candidates = ranked_candidates(global_scores, reference.ids, config["rerank_candidates"])
    ranks, betas, local_ranks, winners, statistics, diagnostics, cached_scores, ties = [], [], [], [], [], [], [], []
    for start in range(0, len(query.ids), config["local_query_batch_size"]):
        indices = torch.arange(start, min(start + config["local_query_batch_size"], len(query.ids)))
        chosen = candidates[indices]
        scores, evidence, transport = score_candidates(model, query, reference, indices, chosen, device)
        g = global_scores[indices[:, None], chosen].to(device)
        if fixed_beta is None:
            fused, beta = model.fuse(g, scores)
        else:
            beta = scores.new_full((len(indices),), fixed_beta)
            fused = (1 - beta[:, None]) * g + beta[:, None] * scores
        fused, scores = fused.cpu(), scores.cpu()
        # 集合内按分数比较，集合外继续保留原全局顺序。
        for row, index in enumerate(indices.tolist()):
            correct = reference.lookup[int(query.ids[index])]
            positions = (chosen[row] == correct).nonzero(as_tuple=True)[0]
            if len(positions):
                position = int(positions[0])
                ranks.append(int((fused[row] > fused[row, position]).sum()))
                local_ranks.append(int((scores[row] > scores[row, position]).sum()))
            else:
                ranks.append(int((global_scores[index] > global_scores[index, correct]).sum()))
                local_ranks.append(None)
            id_order = reference.ids[chosen[row]].argsort()
            winner = id_order[fused[row, id_order].argsort(descending=True, stable=True)[0]]
            winners.append(int(reference.ids[chosen[row, winner]]))
            ties.append(int((fused[row] == fused[row].max()).sum()))
        betas.extend(beta.cpu().tolist())
        statistics.append(evidence.cpu())
        diagnostics.append(transport.cpu())
        cached_scores.append(scores)
    ranks = torch.tensor(ranks)
    correct_indices = torch.tensor([reference.lookup[int(value)] for value in query.ids])
    global_ranks = (global_scores > global_scores[torch.arange(len(query.ids)), correct_indices, None]).sum(1)
    contained = torch.tensor([value is not None for value in local_ranks])
    conditional = [value == 0 for value in local_ranks if value is not None]
    return {"metrics": recall_from_ranks(ranks, len(reference.ids)), "ranks": ranks.tolist(),
            "global_ranks": global_ranks.tolist(), "beta": betas, "top1_ids": winners,
            "top1_ties": ties, "top1_accuracy": sum(a == b for a, b in zip(winners, query.ids.tolist())) / len(winners) * 100,
            "candidate_ids": reference.ids[candidates].tolist(),
            "repaired": int(((global_ranks > 0) & (ranks == 0)).sum()),
            "harmed": int(((global_ranks == 0) & (ranks > 0)).sum()),
            "global_topk_recall": float(contained.double().mean() * 100),
            "conditional_local_accuracy": sum(conditional) / len(conditional) * 100 if conditional else None,
            "statistics": torch.cat(statistics).tolist(), "transport": torch.cat(diagnostics).tolist(),
            "local_scores": torch.cat(cached_scores).tolist(),
            "global_scores": global_scores[torch.arange(len(query.ids))[:, None], candidates].tolist(),
            "local_weight": model.local_weight if fixed_beta is None else fixed_beta}


def fixed_fusion_metrics(result, query, reference, global_scores, beta):
    if not 0 <= beta <= 1:
        raise ValueError("固定局部比例超出范围")
    candidates = torch.tensor([reference.indices(row) for row in result["candidate_ids"]])
    local = torch.tensor(result["local_scores"], dtype=torch.float32)
    indices = torch.arange(len(query.ids))
    fused = (1 - beta) * global_scores[indices[:, None], candidates] + beta * local
    ranks = []
    for row, query_id in enumerate(query.ids.tolist()):
        correct = reference.lookup[query_id]
        positions = (candidates[row] == correct).nonzero(as_tuple=True)[0]
        if len(positions):
            ranks.append(int((fused[row] > fused[row, int(positions[0])]).sum()))
        else:
            ranks.append(int((global_scores[row] > global_scores[row, correct]).sum()))
    return recall_from_ranks(torch.tensor(ranks), len(reference.ids))
