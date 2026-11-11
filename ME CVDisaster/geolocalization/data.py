import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sample4geo.dataset.cvdisaster import CVDisasterDatasetEval, CVDisasterDatasetTrain
from sample4geo.transforms import Cut, get_transforms_train, get_transforms_val
from scripts.prepare_geoloc_data import validate_splits


class RecordedCut(Cut):
    def get_transform_init_args_names(self):
        return ("cutting",)


def recordable_ground_transform(transform):
    original = transform.transforms[0]
    if type(original) is not Cut:
        raise TypeError("街景增强的首个操作必须为作者 Cut")
    # 继承作者的图像处理，仅声明实际存在的序列化参数。
    transform.transforms[0] = RecordedCut(cutting=original.cutting, always_apply=original.always_apply, p=original.p)
    return transform


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_datasets(config, data_config):
    root = config["data_root"]
    manifest = validate_splits(Path(root).resolve())
    if [manifest["splits"][name]["rows"] for name in ("train_60.csv", "val_60.csv")] != [2472, 1649]:
        raise ValueError("实验要求训练 2472 对、测试 1649 对图片")
    sat_val, ground_val = get_transforms_val((384, 384), (140, 768),
                                           mean=data_config["mean"], std=data_config["std"])
    sat_train, ground_train = get_transforms_train((384, 384), (140, 768),
                                                 mean=data_config["mean"], std=data_config["std"])
    ground_train = recordable_ground_transform(ground_train)
    ground_val = recordable_ground_transform(ground_val)
    datasets = {"train": CVDisasterDatasetTrain(
        root, transforms_query=ground_train, transforms_reference=sat_train,
        prob_flip=config["prob_flip"], prob_rotate=config["prob_rotate"],
        shuffle_batch_size=config["batch_size"], split_key="val60")}
    for split in ("train", "test"):
        datasets[split + "_ground"] = CVDisasterDatasetEval(
            root, split, "query", transforms=ground_val, split_key="val60")
        datasets[split + "_satellite"] = CVDisasterDatasetEval(
            root, split, "reference", transforms=sat_val, split_key="val60")
    transforms = {"satellite_train": sat_train.to_dict(), "ground_train": ground_train.to_dict(),
                  "satellite_eval": sat_val.to_dict(), "ground_eval": ground_val.to_dict(),
                  "ground_crop_train": [160, 350], "ground_crop_eval": [160, 360]}
    return datasets, manifest, transforms


def make_loaders(datasets, config, seed):
    loaders = {}
    for name, dataset in datasets.items():
        generator = torch.Generator().manual_seed(seed)
        loaders[name] = DataLoader(
            dataset, batch_size=config["batch_size"] if name == "train" else config["batch_size_eval"],
            shuffle=False, num_workers=config["num_workers"], pin_memory=True,
            worker_init_fn=seed_worker, generator=generator, persistent_workers=False)
    return loaders


def shuffle_training(dataset, neighbours, config):
    # 保留作者的难负采样顺序，同时核验每轮完整 batch 和图片 ID 唯一性。
    dataset.shuffle(neighbours, neighbour_select=config["neighbour_select"],
                    neighbour_range=config["neighbour_range"])
    expected = config["steps_per_epoch"] * config["batch_size"]
    if len(dataset.samples) != expected or len(set(dataset.samples)) != expected:
        raise ValueError(f"本轮采样数量或唯一性错误: {len(dataset.samples)}，要求 {expected}")
    if not set(dataset.samples).issubset(dataset.train_ids):
        raise ValueError("本轮采样包含训练集之外的 ID")
    return [int(value) for value in dataset.samples]
