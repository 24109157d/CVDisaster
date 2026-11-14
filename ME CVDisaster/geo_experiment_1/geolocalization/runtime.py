import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import torch

from . import REPOSITORY
from .model import GeoModel
from sample4geo.utils import setup_system


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_compute_node():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        raise RuntimeError("训练、评估和模型验收必须通过 Slurm 在计算节点运行")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("本实验要求两张可用的 CUDA GPU")
    for index in range(2):
        if "A100" not in torch.cuda.get_device_name(index):
            raise RuntimeError(f"GPU {index} 与方案要求不符: {torch.cuda.get_device_name(index)}")
    torch.cuda.set_device(0)


def initialize(seed):
    # 调用当前基线的后端设置，并保存 PyTorch 实际采用的设置。
    setup_system(seed, cudnn_benchmark=True, cudnn_deterministic=False)
    return {"seed": seed, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def new_output(path):
    path = Path(path).resolve()
    if path.is_relative_to(REPOSITORY):
        raise ValueError("运行结果必须保存在仓库之外")
    path.mkdir(parents=True, exist_ok=False)
    return path


def create_model(config, architecture):
    return GeoModel(architecture, config["transport_tau"], config["sinkhorn_iterations"], config["module_init_seed"],
                    config["fusion_weight_init"], config["sinkhorn_max_iterations"],
                    config["sinkhorn_check_interval"], config["sinkhorn_tolerance"])


def create_optimizer(model, config):
    groups = {"geo": [], "spatial": [], "fusion": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "fusion_logit":
            group = "fusion"
        elif name.partition(".")[0] in ("channel_projection", "feature_transport", "refinement"):
            group = "spatial"
        else:
            group = "geo"
        groups[group].append(parameter)
    parameters = [{"name": name, "params": values,
                   "lr": config["lr"] if name == "geo" else config[name + "_lr"],
                   "weight_decay": config["fusion_weight_decay"] if name == "fusion" else config["weight_decay"]}
                  for name, values in groups.items() if values]
    return torch.optim.AdamW(parameters, lr=config["lr"], betas=tuple(config["betas"]),
                             eps=config["eps"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])


def load_initial_model(config, architecture):
    actual = sha256(config["cvusa_checkpoint"])
    if actual != config["cvusa_sha256"]:
        raise ValueError(f"CVUSA 权重 SHA256 不符: {actual}")
    model = create_model(config, architecture)
    scales = model.load_cvusa(config["cvusa_checkpoint"])
    return model, scales


class GeoDataParallel(torch.nn.DataParallel):
    def forward(self, ground, satellite=None, sample_ids=None, view=None):
        if self.module.architecture == "baseline":
            return super().forward(ground, satellite, sample_ids=sample_ids, view=view)
        # 汇集全部样本后求解搬移矩阵，整个 batch 使用同一个停止条件。
        encoded = super().forward(ground, satellite, sample_ids=sample_ids, view=view, defer_spatial=True)
        if satellite is not None:
            return (self.module.complete_encoding(encoded[0], "ground", sample_ids),
                    self.module.complete_encoding(encoded[1], "satellite", sample_ids))
        return self.module.complete_encoding(encoded, view, sample_ids)


def parallel_model(model):
    return GeoDataParallel(model.cuda(), device_ids=[0, 1])


def environment():
    commit = os.environ["CVDISASTER_COMMIT"]
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    return {"commit": commit, "python": sys.version, "platform": platform.platform(),
            "packages": packages, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
            "job_id": os.environ["SLURM_JOB_ID"], "node": os.environ["SLURMD_NODENAME"],
            "argv": sys.argv, "slurm_script": os.environ.get("CVDISASTER_SLURM_SCRIPT")}


def memory_usage():
    import resource

    return {"gpu_peak_allocated": [torch.cuda.max_memory_allocated(i) for i in range(2)],
            "gpu_peak_reserved": [torch.cuda.max_memory_reserved(i) for i in range(2)],
            "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "workers_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
