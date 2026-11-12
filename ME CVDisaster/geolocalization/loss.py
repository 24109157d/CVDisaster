from torch import nn

from sample4geo.loss import InfoNCE


class DescriptorInfoNCE(InfoNCE):
    def forward(self, ground, satellite, logit_scale):
        return super().forward(ground["descriptors"], satellite["descriptors"], logit_scale)


def contrastive_loss(ground, satellite, model, sample_ids, smoothing=0.1):
    criterion = DescriptorInfoNCE(nn.CrossEntropyLoss(label_smoothing=smoothing), device=sample_ids.device)
    return criterion(ground, satellite, model.logit_scale.exp())
