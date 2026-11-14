import math

from torch import nn

from sample4geo.loss import InfoNCE


class DescriptorInfoNCE(InfoNCE):
    def __init__(self, loss_function, device="cuda", weights=None):
        super().__init__(loss_function, device=device)
        self.weights = dict(weights if weights is not None else {"global": 1.0, "spatial": 1.0, "fusion": 1.0})
        if set(self.weights) != {"global", "spatial", "fusion"} or any(
                not math.isfinite(value) or value <= 0 for value in self.weights.values()):
            raise ValueError("三个分支损失系数必须为有限正数")
        self.components = {}

    def forward(self, ground, satellite, logit_scale):
        if "spatial_features" not in ground:
            loss = super().forward(ground["descriptors"], satellite["descriptors"], logit_scale)
            self.components = {"global": loss.detach()}
            return loss
        losses = {name: super(DescriptorInfoNCE, self).forward(ground[key], satellite[key], logit_scale)
                  for name, key in (("global", "global_raw"), ("spatial", "spatial_features"),
                                    ("fusion", "descriptors"))}
        self.components = {name: value.detach() for name, value in losses.items()}
        return sum(self.weights[name] * value for name, value in losses.items())


def contrastive_loss(ground, satellite, model, sample_ids, smoothing=0.1, weights=None):
    criterion = DescriptorInfoNCE(nn.CrossEntropyLoss(label_smoothing=smoothing), device=sample_ids.device,
                                 weights=weights)
    return criterion(ground, satellite, model.logit_scale.exp())
