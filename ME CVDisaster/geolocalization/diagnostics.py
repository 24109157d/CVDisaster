import torch


def gradient_statistics(model):
    groups = {}
    missing, nonfinite = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = name.partition(".")[0]
        groups.setdefault(group, 0.0)
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
        else:
            groups[group] = max(groups[group], float(parameter.grad.detach().abs().max()))
    return {"max_abs_finite": groups, "missing": missing, "nonfinite": nonfinite}
