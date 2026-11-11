import torch
from torch import nn
from torch.cuda.amp import autocast


def contrastive_loss(ground, satellite, model, sample_ids, smoothing=0.1):
    if sample_ids.ndim != 1 or len(sample_ids.unique()) != len(sample_ids):
        raise ValueError(f"训练 batch 图片 ID 必须唯一: {sample_ids.tolist()}")
    labels = torch.arange(len(sample_ids), device=sample_ids.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing, reduction="mean")
    with autocast(enabled=False):
        first, second = ground["descriptors"].float(), satellite["descriptors"].float()
        if first.shape != second.shape or len(first) != len(sample_ids):
            raise ValueError("两张 GPU 汇集后的最终向量与全局 batch 不一致")
        logits = model.logit_scale.float().exp() * (first @ second.T)
        if logits.shape != (len(labels), len(labels)) or not torch.isfinite(logits).all():
            raise FloatingPointError("最终向量的 InfoNCE logits 形状或数值错误")
        loss = (criterion(logits, labels) + criterion(logits.T, labels)) / 2
    if not torch.isfinite(loss):
        raise FloatingPointError(f"训练损失包含非有限数值: sample_ids={sample_ids.tolist()}")
    return loss
