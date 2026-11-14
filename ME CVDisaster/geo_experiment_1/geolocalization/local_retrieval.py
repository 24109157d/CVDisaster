import random

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
def mine_local_candidates(model, query, reference, global_scores, config, device, seed):
    model.eval()
    k = config["rerank_candidates"]
    top = ranked_candidates(global_scores, reference.ids, k)
    local = []
    for start in range(0, len(query.ids), config["local_query_batch_size"]):
        selected = torch.arange(start, min(start + config["local_query_batch_size"], len(query.ids)))
        scores, _, _ = score_candidates(model, query, reference, selected, top[selected], device)
        local.append(scores.cpu())
    local = torch.cat(local)
    rng = random.Random(seed)
    result = []
    for index, query_id in enumerate(query.ids.tolist()):
        correct = reference.lookup[query_id]
        selected = [correct]
        global_order = top[index].tolist()
        if config["global_hard_count"]:
            for candidate in global_order:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) == 1 + config["global_hard_count"]:
                    break
        target = len(selected) + config["local_hard_count"]
        if config["local_hard_count"]:
            for position in local[index].argsort(descending=True, stable=True).tolist():
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
    return torch.tensor(result, dtype=torch.long), top


def direction_losses(model, query, reference, indices, local_candidates, fusion_candidates,
                     global_scores, scale, config, device):
    selected = torch.as_tensor(indices, dtype=torch.long)
    local_scores, _, diagnostics = score_candidates(model, query, reference, selected,
                                                     local_candidates[selected], device)
    labels = torch.zeros(len(selected), dtype=torch.long, device=device)
    scale = scale.detach().float()
    local_loss = F.cross_entropy(scale * local_scores, labels, label_smoothing=config["label_smoothing"])
    candidates = fusion_candidates[selected]
    correct_ids = query.ids[selected]
    reference_ids = reference.ids[candidates]
    matches = reference_ids == correct_ids[:, None]
    valid = matches.any(1)
    if valid.any():
        valid_indices, candidates = selected[valid], candidates[valid]
        scores, statistics, more_diagnostics = score_candidates(model, query, reference, valid_indices, candidates, device)
        g_scores = global_scores[valid_indices[:, None], candidates].to(device)
        fused, beta = model.fuse(g_scores, scores, statistics)
        targets = matches[valid].long().argmax(1).to(device)
        fusion_loss = F.cross_entropy(scale * fused, targets, label_smoothing=config["label_smoothing"])
        diagnostics = torch.cat((diagnostics, more_diagnostics))
        beta_mean = float(beta.detach().mean())
    else:
        fusion_loss = local_scores.sum() * 0
        beta_mean = None
    return local_loss, fusion_loss, {"valid_fusion_queries": int(valid.sum()), "queries": len(selected),
                                      "beta_mean": beta_mean, "diagnostics": diagnostics.detach().cpu()}


def recall_from_ranks(ranks, gallery_size):
    return {**{f"R@{k}": float((ranks < k).double().mean() * 100) for k in (1, 5, 10)},
            "R@1%": float((ranks < gallery_size // 100).double().mean() * 100)}


@torch.no_grad()
def rerank(model, query, reference, global_scores, config, device, fixed_beta=None):
    model.eval()
    candidates = ranked_candidates(global_scores, reference.ids, config["rerank_candidates"])
    ranks, betas, local_ranks, winners, statistics, diagnostics, cached_scores, ties = [], [], [], [], [], [], [], []
    for start in range(0, len(query.ids), config["local_query_batch_size"]):
        indices = torch.arange(start, min(start + config["local_query_batch_size"], len(query.ids)))
        chosen = candidates[indices]
        scores, evidence, transport = score_candidates(model, query, reference, indices, chosen, device)
        g = global_scores[indices[:, None], chosen].to(device)
        if fixed_beta is None:
            fused, beta = model.fuse(g, scores, evidence)
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
            "local_scores": torch.cat(cached_scores).tolist()}


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
