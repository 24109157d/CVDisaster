import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from sample4geo.model import TimmModel
from .spatial import create_spatial_modules


def normalize_features(features, name):
    features = features.float()
    norms = features.norm(dim=-1, keepdim=True)
    if not torch.isfinite(features).all() or not torch.isfinite(norms).all() or (norms == 0).any():
        raise FloatingPointError(f"{name} 包含非有限数值或零范数")
    return F.normalize(features, p=2, dim=-1, eps=1e-12)


class GeoModel(TimmModel):
    def __init__(self, architecture, transport_tau=1.0, sinkhorn_iterations=30, module_init_seed=42):
        if architecture not in ("baseline", "geo_cvft_refine"):
            raise ValueError(f"未知模型: {architecture}")
        # 正式运行从已有 CVUSA 权重加载全部原有参数，不触发在线下载。
        super().__init__("convnext_base.fb_in22k_ft_in1k_384", pretrained=False, img_size=384)
        self.architecture = architecture
        if architecture == "geo_cvft_refine":
            self.fusion_logit = nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.ground_pool = nn.AdaptiveAvgPool2d((4, 16))
            self.satellite_pool = nn.AdaptiveAvgPool2d((8, 8))
            self.channel_projection, self.feature_transport, self.refinement = create_spatial_modules(
                transport_tau, sinkhorn_iterations, module_init_seed)

    def load_cvusa(self, checkpoint):
        state = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state, strict=False)
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
        features = self.model.forward_features(image)
        global_raw = self.model.forward_head(features)
        if self.architecture == "baseline":
            return {"global_features": F.normalize(global_raw, dim=-1), "descriptors": global_raw}
        expected_shape = (1024, 4, 24) if view == "ground" else (1024, 12, 12)
        if tuple(features.shape[1:]) != expected_shape:
            raise ValueError(f"{view} 空间特征形状错误: {tuple(features.shape)}")
        with autocast(enabled=False):
            global_features = normalize_features(global_raw, "global_features")
        result = {"global_features": global_features, "descriptors": global_features}
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
            result["descriptors"] = fused
            # 每个样本携带同一个比例，DataParallel 汇集时保留图片顺序。
            result["fusion_weight"] = alpha.expand(len(image))
        return result

    def forward(self, ground, satellite=None, sample_ids=None, view=None):
        if satellite is not None:
            return self.encode(ground, "ground", sample_ids), self.encode(satellite, "satellite", sample_ids)
        return self.encode(ground, view, sample_ids)
