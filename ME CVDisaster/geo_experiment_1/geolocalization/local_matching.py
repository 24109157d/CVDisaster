import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class AttentionLayer(nn.Module):
    def __init__(self, dimension, heads, hidden):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dimension, eps=1e-5)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=0.0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dimension, eps=1e-5)
        self.ffn = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, dimension))

    def forward(self, ground, satellite, cross):
        g, s = self.attention_norm(ground), self.attention_norm(satellite)
        g_source, s_source = (s, g) if cross else (g, s)
        ground = ground + self.attention(g, g_source, g_source, need_weights=False)[0]
        satellite = satellite + self.attention(s, s_source, s_source, need_weights=False)[0]
        return ground + self.ffn(self.ffn_norm(ground)), satellite + self.ffn(self.ffn_norm(satellite))


class PartialTransport(nn.Module):
    def __init__(self, temperature, minimum, maximum, interval, tolerance):
        super().__init__()
        if temperature <= 0 or minimum < 1 or maximum < minimum or interval < 1 or tolerance <= 0:
            raise ValueError("部分匹配参数不符合范围")
        if (maximum - minimum) % interval:
            raise ValueError("最大迭代次数必须处于检查序列中")
        self.temperature = temperature
        self.minimum, self.maximum, self.interval, self.tolerance = minimum, maximum, interval, tolerance
        self.bin_score = nn.Parameter(torch.tensor(1.0))

    def forward(self, ground, satellite):
        with torch.autocast(device_type=ground.device.type, enabled=False):
            ground, satellite = ground.float(), satellite.float()
            scores = ground @ satellite.transpose(1, 2) / self.temperature
            batch, m, n = scores.shape
            if min(m, n) < 2 or not torch.isfinite(scores).all():
                raise ValueError("位置数量不足或匹配分数包含非有限值")
            bins = self.bin_score.float()
            logits = torch.cat((torch.cat((scores, bins.expand(batch, m, 1)), 2),
                                bins.expand(batch, 1, n + 1)), 1)
            a = scores.new_ones(m + 1)
            b = scores.new_ones(n + 1)
            a[-1], b[-1] = n, m
            normalizer = math.log(m + n)
            log_a, log_b = a.log() - normalizer, b.log() - normalizer
            u = torch.zeros_like(log_a).expand(batch, -1)
            v = torch.zeros_like(log_b).expand(batch, -1)
            for iteration in range(1, self.maximum + 1):
                u = log_a - torch.logsumexp(logits + v[:, None, :], dim=2)
                v = log_b - torch.logsumexp(logits + u[:, :, None], dim=1)
                if iteration >= self.minimum and (iteration - self.minimum) % self.interval == 0:
                    with torch.no_grad():
                        checked = (logits + u[:, :, None] + v[:, None, :] + normalizer).exp()
                        row = (checked.sum(2) - a).abs().amax(1)
                        column = (checked.sum(1) - b).abs().amax(1)
                        error = torch.maximum(row, column)
                        if not torch.isfinite(error).all():
                            raise FloatingPointError("部分匹配误差包含非有限值")
                        if (error <= self.tolerance).all():
                            break
            assignment = (logits + u[:, :, None] + v[:, None, :] + normalizer).exp()
            if not torch.isfinite(assignment).all():
                raise FloatingPointError("部分匹配矩阵包含非有限值")
            diagnostics = torch.stack((row, column, torch.full_like(row, iteration),
                                       (error <= self.tolerance).float()), 1)
            return assignment[:, :m, :n], diagnostics


def matching_evidence(ground, satellite, weights):
    m, n = weights.shape[1:]
    rows, columns = weights.sum(2), weights.sum(1)
    total = rows.sum(1)
    conditional_rows = weights / (rows[:, :, None] + 1e-8)
    conditional_columns = weights / (columns[:, None, :] + 1e-8)
    row_entropy = -(conditional_rows * (conditional_rows + 1e-8).log()).sum(2)
    column_entropy = -(conditional_columns * (conditional_columns + 1e-8).log()).sum(1)
    statistics = torch.stack((total / min(m, n), weights.amax(2).mean(1), weights.amax(1).mean(1),
                              (rows * row_entropy).sum(1) / ((total + 1e-8) * math.log(n)),
                              (columns * column_entropy).sum(1) / ((total + 1e-8) * math.log(m)),
                              (conditional_rows * conditional_columns).sum((1, 2)) / min(m, n)), 1)
    difference = ((ground[:, :, None, :] - satellite[:, None, :, :]).abs()
                  * weights[:, :, :, None]).sum((1, 2)) / min(m, n)
    product = (ground * (weights @ satellite)).sum(1) / min(m, n)
    evidence = torch.cat((difference, product, statistics), 1)
    if not torch.isfinite(evidence).all():
        raise FloatingPointError("局部匹配证据包含非有限值")
    return evidence, statistics


class LocalMatcher(nn.Module):
    def __init__(self, config):
        super().__init__()
        dimension = config["local_dimension"]
        self.dimension = dimension
        self.pair_group_size = config["pair_group_size"]
        self.checkpointing = config["activation_checkpointing"]
        self.projection2 = nn.Conv2d(512, dimension // 2, 1)
        self.projection3 = nn.Conv2d(1024, dimension // 2, 1)
        self.position = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, dimension))
        self.view_embedding = nn.Parameter(torch.empty(2, dimension))
        self.layers = nn.ModuleList([AttentionLayer(dimension, config["attention_heads"], config["ffn_hidden"])
                                     for _ in range(2 * config["attention_blocks"])])
        self.final_projection = nn.Linear(dimension, dimension)
        self.transport = PartialTransport(config["match_temperature"], config["sinkhorn_iterations"],
                                          config["sinkhorn_max_iterations"], config["sinkhorn_check_interval"],
                                          config["sinkhorn_tolerance"])
        self.scoring = nn.Sequential(nn.Linear(2 * dimension + 6, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
        self.local_weight = float(config["local_weight"])
        if not 0 <= self.local_weight <= 1 or dimension % 2 or dimension % config["attention_heads"] or self.pair_group_size < 1:
            raise ValueError("局部模块形状或初始化参数不符合范围")
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.normal_(self.view_embedding, std=0.02)

    def encode(self, stage2, stage3, view):
        if view not in ("ground", "satellite") or stage2.shape[1] != 512 or stage3.shape[1] != 1024:
            raise ValueError("局部输入视角或通道数量错误")
        stage2, stage3 = stage2.detach(), stage3.detach()
        stage2 = F.adaptive_avg_pool2d(stage2, stage3.shape[-2:])
        features = torch.cat((self.projection2(stage2), self.projection3(stage3)), 1).flatten(2).transpose(1, 2)
        height, width = stage3.shape[-2:]
        y = 2 * (torch.arange(height, device=stage3.device, dtype=torch.float32) + 0.5) / height - 1
        x = 2 * (torch.arange(width, device=stage3.device, dtype=torch.float32) + 0.5) / width - 1
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        positions = torch.stack((xx, yy), -1).reshape(-1, 2)
        return features + self.position(positions)[None] + self.view_embedding[0 if view == "ground" else 1]

    def match(self, ground, satellite):
        for index, layer in enumerate(self.layers):
            ground, satellite = layer(ground, satellite, cross=index % 2 == 1)
        with torch.autocast(device_type=ground.device.type, enabled=False):
            ground = F.normalize(self.final_projection(ground.float()), dim=-1, eps=1e-8)
            satellite = F.normalize(self.final_projection(satellite.float()), dim=-1, eps=1e-8)
            weights, diagnostics = self.transport(ground, satellite)
            evidence, statistics = matching_evidence(ground, satellite, weights)
            scores = self.scoring(evidence).squeeze(1)
            if not torch.isfinite(scores).all():
                raise FloatingPointError("局部分数包含非有限值")
        return scores, statistics, diagnostics

    def score_pairs(self, ground, satellite):
        if len(ground) != len(satellite) or len(ground) == 0:
            raise ValueError("候选对数量不一致或为空")
        groups = []
        for start in range(0, len(ground), self.pair_group_size):
            g = ground[start:start + self.pair_group_size]
            s = satellite[start:start + self.pair_group_size]
            if self.training and self.checkpointing and torch.is_grad_enabled():
                if not g.requires_grad or not s.requires_grad:
                    raise RuntimeError("checkpoint 输入必须保留新增投影的梯度")
                result = checkpoint(self.match, g, s, preserve_rng_state=True, use_reentrant=True)
            else:
                result = self.match(g, s)
            groups.append(result)
        return tuple(torch.cat([group[index] for group in groups]) for index in range(3))

    def fuse(self, global_scores, local_scores):
        if global_scores.shape != local_scores.shape or global_scores.shape[1] < 2:
            raise ValueError("融合候选分数形状错误")
        global_scores = global_scores.detach().float()
        local_scores = local_scores.float()
        with torch.autocast(device_type=global_scores.device.type, enabled=False):
            beta = local_scores.new_full((len(local_scores),), self.local_weight)
            scores = (1 - self.local_weight) * global_scores + self.local_weight * local_scores
        return scores, beta


def create_local_optimizer(model, config):
    groups = {"local": [], "bin": []}
    for name, parameter in model.named_parameters():
        group = "bin" if name == "transport.bin_score" else "local"
        groups[group].append(parameter)
    return torch.optim.AdamW([
        {"name": name, "params": values,
         "lr": config["local_lr"],
         "weight_decay": 0.0 if name == "bin" else config["weight_decay"]}
        for name, values in groups.items()], betas=tuple(config["betas"]), eps=config["eps"], amsgrad=False)
