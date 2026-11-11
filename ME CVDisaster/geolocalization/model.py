import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from sample4geo.model import TimmModel


def normalize_features(features, name):
    features = features.float()
    norms = features.norm(dim=-1, keepdim=True)
    if not torch.isfinite(features).all() or not torch.isfinite(norms).all() or (norms == 0).any():
        raise FloatingPointError(f"{name} 包含非有限数值或零范数")
    return F.normalize(features, p=2, dim=-1, eps=1e-12)


class FeatureTransport(nn.Module):
    def __init__(self, tau=1.0, iterations=20, tolerance=1e-3):
        super().__init__()
        if tau <= 0 or iterations < 1 or tolerance <= 0:
            raise ValueError("Sinkhorn 参数必须为正数")
        self.tau = tau
        self.iterations = iterations
        self.tolerance = tolerance
        self.compress = nn.Conv2d(64, 1, 1, bias=True)
        self.activation = nn.GELU()
        self.cost = nn.Linear(64, 4096, bias=True)

    def forward(self, ground, sample_ids):
        # 对数域归一化和矩阵乘法全部使用 FP32，梯度继续传入街景编码器。
        with autocast(enabled=False):
            ground = ground.float()
            cost = self.cost(self.activation(self.compress(ground)).flatten(1)).reshape(-1, 64, 64)
            log_p = -cost / self.tau
            for _ in range(self.iterations):
                log_p = log_p - torch.logsumexp(log_p, dim=2, keepdim=True)
                log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
            transport = log_p.exp()
            row_error = (transport.sum(2) - 1).abs().amax(1)
            column_error = (transport.sum(1) - 1).abs().amax(1)
            valid = (torch.isfinite(cost).all(dim=2).all(dim=1)
                     & torch.isfinite(transport).all(dim=2).all(dim=1)
                     & (transport >= 0).all(dim=2).all(dim=1)
                     & (row_error <= self.tolerance) & (column_error <= self.tolerance))
            if not valid.all():
                failed = (~valid).nonzero(as_tuple=True)[0]
                raise FloatingPointError(
                    f"Sinkhorn 检查失败: sample_ids={sample_ids[failed].tolist()}, "
                    f"cost_min={cost[failed].amin(dim=(1, 2)).tolist()}, "
                    f"cost_max={cost[failed].amax(dim=(1, 2)).tolist()}, "
                    f"e_row={row_error[failed].tolist()}, e_col={column_error[failed].tolist()}, "
                    f"transport_tau={self.tau}")
            # G 的位置顺序为 row*16+column，P 的行表示源位置，列表示目标位置。
            source = ground.flatten(2).transpose(1, 2)
            moved = transport.transpose(1, 2) @ source
            moved = moved.transpose(1, 2).reshape(-1, 64, 8, 8)
            entropy = -(transport * (transport + 1e-12).log()).sum((1, 2)) / 64
            diagnostics = torch.stack((entropy, transport.amax((1, 2)), row_error, column_error), dim=1)
        return moved, transport, diagnostics


class SpatialRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64, eps=1e-6)
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=True)

    def forward(self, features):
        # LayerNorm 在每个位置的通道维度计算；卷积使用 NCHW 顺序。
        normalized = self.norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return features + self.conv2(self.activation(self.conv1(normalized)))


class GeoModel(TimmModel):
    def __init__(self, architecture, transport_tau=1.0, sinkhorn_iterations=20,
                 sinkhorn_tolerance=1e-3):
        if architecture not in ("baseline", "geo_cvft_refine"):
            raise ValueError(f"未知模型: {architecture}")
        # 正式运行从已有 CVUSA 权重加载全部原有参数，不触发在线下载。
        super().__init__("convnext_base.fb_in22k_ft_in1k_384", pretrained=False, img_size=384)
        self.architecture = architecture
        if architecture == "geo_cvft_refine":
            self.fusion_logit = nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.channel_projection = nn.Conv2d(1024, 64, 1, bias=True)
            self.ground_pool = nn.AdaptiveAvgPool2d((4, 16))
            self.satellite_pool = nn.AdaptiveAvgPool2d((8, 8))
            self.feature_transport = FeatureTransport(transport_tau, sinkhorn_iterations, sinkhorn_tolerance)
            self.refinement = SpatialRefinement()
            for module in (self.channel_projection, self.feature_transport, self.refinement):
                for layer in module.modules():
                    if isinstance(layer, (nn.Conv2d, nn.Linear)):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)

    def load_cvusa(self, checkpoint):
        state = torch.load(checkpoint, map_location="cpu")
        current = self.state_dict()
        expected = {name for name in current if name.startswith("model.") or name == "logit_scale"}
        if set(state) != expected:
            raise ValueError(f"CVUSA 参数不完整: missing={sorted(expected - set(state))}, "
                             f"unexpected={sorted(set(state) - expected)}")
        current.update(state)
        self.load_state_dict(current, strict=True)
        return {"logit_scale": float(self.logit_scale.detach())}

    def fusion_coefficients(self):
        with autocast(enabled=False):
            value = self.fusion_logit.float()
            alpha = value.sigmoid()
            global_scale = (0.5 * F.logsigmoid(value)).exp()
            spatial_scale = (0.5 * F.logsigmoid(-value)).exp()
            if not torch.isfinite(torch.stack((value, alpha, global_scale, spatial_scale))).all():
                raise FloatingPointError("融合参数或缩放系数包含非有限数值")
        return alpha, global_scale, spatial_scale

    def fusion_state(self):
        if self.architecture == "baseline":
            return {"fusion_logit": None, "fusion_weight": 1.0}
        alpha, global_scale, spatial_scale = self.fusion_coefficients()
        return {"fusion_logit": float(self.fusion_logit.detach()), "fusion_weight": float(alpha.detach()),
                "global_scale": float(global_scale.detach()), "spatial_scale": float(spatial_scale.detach())}

    def encode(self, image, view, sample_ids):
        if view not in ("ground", "satellite"):
            raise ValueError("view 必须为 ground 或 satellite")
        expected_input = (140, 768) if view == "ground" else (384, 384)
        if tuple(image.shape[1:]) != (3, *expected_input):
            raise ValueError(f"{view} 输入形状错误: {tuple(image.shape)}")
        features = self.model.forward_features(image)
        expected_shape = (1024, 4, 24) if view == "ground" else (1024, 12, 12)
        if tuple(features.shape[1:]) != expected_shape:
            raise ValueError(f"{view} 空间特征形状错误: {tuple(features.shape)}")
        global_raw = self.model.forward_head(features)
        with autocast(enabled=False):
            global_features = normalize_features(global_raw, "global_features")
        result = {"global_features": global_features, "descriptors": global_features}
        if self.architecture == "baseline":
            return result
        projected = self.channel_projection(features)
        if view == "ground":
            if sample_ids is None or len(sample_ids) != len(image):
                raise ValueError("街景前向计算需要对应的 sample_ids")
            projected = self.ground_pool(projected)
            moved, transport, diagnostics = self.feature_transport(projected, sample_ids)
            spatial_map = self.refinement(moved)
            result.update(transport=transport, transport_diagnostics=diagnostics)
        else:
            spatial_map = self.satellite_pool(projected)
        with autocast(enabled=False):
            # 两种视角均按目标位置、通道顺序展开，位置索引为 row*8+column。
            spatial = spatial_map.float().flatten(2).transpose(1, 2).reshape(len(image), 4096)
            spatial = normalize_features(spatial, "spatial_features")
            result["spatial_features"] = spatial
            alpha, global_scale, spatial_scale = self.fusion_coefficients()
            fused = torch.cat((global_scale * global_features, spatial_scale * spatial), dim=1)
            result["descriptors"] = normalize_features(fused, "descriptors")
            # 每个样本携带同一个比例，DataParallel 汇集时保留图片顺序。
            result["fusion_weight"] = alpha.expand(len(image))
        return result

    def forward(self, ground, satellite=None, sample_ids=None, view=None):
        if satellite is not None:
            return self.encode(ground, "ground", sample_ids), self.encode(satellite, "satellite", sample_ids)
        return self.encode(ground, view, sample_ids)
