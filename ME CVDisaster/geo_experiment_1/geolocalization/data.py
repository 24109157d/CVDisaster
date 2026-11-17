import hashlib
from pathlib import Path

from torch.utils.data import DataLoader

from sample4geo.dataset.cvdisaster import CVDisasterDatasetEval, CVDisasterDatasetTrain
from sample4geo.transforms import Cut, get_transforms_train, get_transforms_val


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


def make_datasets(config, data_config):
    root = config["data_root"]
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
    manifest = {"data_root": str(Path(root).resolve()), "splits": {}}
    for name, dataset in (("train_60.csv", datasets["train"]), ("val_60.csv", datasets["test_ground"])):
        manifest["splits"][name] = {"rows": len(dataset.df),
                                      "sha256": hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()}
    transforms = {"satellite_train": sat_train.to_dict(), "ground_train": ground_train.to_dict(),
                  "satellite_eval": sat_val.to_dict(), "ground_eval": ground_val.to_dict(),
                  "ground_crop_train": [160, 350], "ground_crop_eval": [160, 360]}
    return datasets, manifest, transforms


def make_loaders(datasets, config):
    loaders = {}
    for name, dataset in datasets.items():
        loaders[name] = DataLoader(
            dataset, batch_size=config["batch_size"] if name == "train" else config["batch_size_eval"],
            shuffle=False, num_workers=config["num_workers"], pin_memory=True)
    return loaders


def shuffle_training(dataset, neighbours, config):
    dataset.shuffle(neighbours, neighbour_select=config["neighbour_select"],
                    neighbour_range=config["neighbour_range"])
    return [int(value) for value in dataset.samples]
