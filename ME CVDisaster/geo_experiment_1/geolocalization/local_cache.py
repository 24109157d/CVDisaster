from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.nn import functional as F

from sample4geo.trainer import predict as author_predict


class IsolatedRandomState:
    def __enter__(self):
        self.python = random.getstate()
        self.numpy = np.random.get_state()
        self.torch = torch.random.fork_rng(devices=list(range(torch.cuda.device_count())))
        self.torch.__enter__()
        return self

    def __exit__(self, exception_type, exception, traceback):
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        return self.torch.__exit__(exception_type, exception, traceback)


@dataclass
class FeatureCache:
    ids: torch.Tensor
    global_features: torch.Tensor
    stage2: torch.Tensor
    stage3: torch.Tensor
    view: str
    context: str

    def __post_init__(self):
        if self.view not in ("ground", "satellite") or not self.context:
            raise ValueError("缓存视角或 checkpoint 标识错误")
        size = len(self.ids)
        if len(set(self.ids.tolist())) != size or any(len(value) != size for value in (
                self.global_features, self.stage2, self.stage3)):
            raise ValueError("缓存图片 ID 重复或特征数量不一致")
        for value in (self.global_features, self.stage2, self.stage3):
            if value.device.type != "cpu" or value.dtype != torch.float32 or value.requires_grad:
                raise ValueError("缓存必须为没有梯度的 CPU FP32 特征")
            if not torch.isfinite(value).all():
                raise FloatingPointError("缓存包含非有限值")
        self.lookup = {int(value): index for index, value in enumerate(self.ids.tolist())}

    def select(self, indices, device):
        indices = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        return self.stage2[indices].to(device), self.stage3[indices].to(device)

    def indices(self, ids):
        return [self.lookup[int(value)] for value in ids]


class StageRecorder:
    def __init__(self, model, view):
        self.model = model
        self.base = model.module if isinstance(model, torch.nn.DataParallel) else model
        self.target = (4, 24) if view == "ground" else (12, 12)
        self.active = False
        self.parts = {}

    def __enter__(self):
        self.handles = []
        for stage in (2, 3):
            def record(module, inputs, output, index=stage):
                if self.active:
                    device = output.device.index
                    if device in self.parts[index]:
                        raise RuntimeError("同一缓存 batch 的 stage 被重复执行")
                    value = F.adaptive_avg_pool2d(output.detach().float(), self.target) if index == 2 else output.detach().float()
                    self.parts[index][device] = value.cpu()
            self.handles.append(self.base.model.stages[stage].register_forward_hook(record))
        return self

    def forward(self, images):
        self.parts = {2: {}, 3: {}}
        self.active = True
        raw = self.model(images)
        self.active = False
        keys = sorted(self.parts[3], key=lambda value: -1 if value is None else value)
        if not keys or set(keys) != set(self.parts[2]):
            raise RuntimeError("全局前向没有产生完整的两层缓存")
        values = [torch.cat([self.parts[index][key] for key in keys]) for index in (2, 3)]
        if any(len(value) != len(images) for value in values):
            raise RuntimeError("DataParallel 缓存顺序或 batch 数量错误")
        return raw, values

    def __exit__(self, exception_type, exception, traceback):
        self.active = False
        for handle in self.handles:
            handle.remove()
        return False


def predict_cached(config, model, loader, view, context):
    collected = {2: [], 3: []}
    with StageRecorder(model, view) as recorder:
        def extract(current_model, images, ids):
            if current_model is not model:
                raise RuntimeError("全局预测模型实例发生变化")
            raw, values = recorder.forward(images)
            for stage, value in zip((2, 3), values):
                collected[stage].append(value)
            return raw
        features, ids = author_predict(config, model, loader, feature_extractor=extract)
    cache = FeatureCache(ids.detach().cpu(), features.detach().cpu(),
                         torch.cat(collected[2]), torch.cat(collected[3]), view, context)
    return cache, features, ids


def ranked_candidates(scores, ids, count):
    if scores.ndim != 2 or scores.shape[1] != len(ids) or not 1 < count <= len(ids):
        raise ValueError("候选数量或分数矩阵形状错误")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("候选分数包含非有限值")
    id_order = ids.argsort()
    return id_order[scores[:, id_order].argsort(dim=1, descending=True, stable=True)[:, :count]]
