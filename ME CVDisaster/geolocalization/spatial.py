import torch
from torch import nn
from torch.cuda.amp import autocast


def initialize_cvft_conv(layer):
    nn.init.xavier_uniform_(layer.weight)
    # CVFT 对卷积 bias 也使用 Xavier；一维参数的 fan_in 和 fan_out 均为通道数。
    bound = (3.0 / layer.bias.numel()) ** 0.5
    nn.init.uniform_(layer.bias, -bound, bound)


def create_spatial_modules(tau, iterations, seed):
    # 新模块在 CPU 初始化；退出后恢复原有 CPU 随机状态，CUDA 随机状态保持原值。
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        projection = nn.Conv2d(1024, 64, 1, bias=True)
        transport = FeatureTransport(tau, iterations)
        refinement = SpatialRefinement()
        initialize_cvft_conv(projection)
    return projection, transport, refinement


class FeatureTransport(nn.Module):
    def __init__(self, tau=1.0, iterations=30):
        super().__init__()
        if tau <= 0 or iterations < 1:
            raise ValueError("Sinkhorn 参数必须为正数")
        self.tau = tau
        self.iterations = iterations
        self.compress = nn.Conv2d(64, 1, 1, bias=True)
        self.activation = nn.ReLU()
        self.cost = nn.Linear(64, 4096, bias=True)
        initialize_cvft_conv(self.compress)
        nn.init.trunc_normal_(self.cost.weight, mean=0.0, std=0.005, a=-0.01, b=0.01)
        nn.init.eye_(self.cost.bias.view(64, 64))

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
                     & torch.isfinite(row_error) & torch.isfinite(column_error))
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
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.norm1 = nn.InstanceNorm2d(64, eps=1e-5, affine=False, track_running_stats=False)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.norm2 = nn.InstanceNorm2d(64, eps=1e-5, affine=False, track_running_stats=False)
        self.activation = nn.ReLU(inplace=True)
        for layer in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, features):
        residual = self.activation(self.norm1(self.conv1(features)))
        residual = self.activation(self.norm2(self.conv2(residual)))
        return self.activation(features + residual)
