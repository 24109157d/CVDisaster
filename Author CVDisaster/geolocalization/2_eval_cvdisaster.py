import os
import argparse
import torch
from dataclasses import dataclass

from torch.utils.data import DataLoader
from sample4geo.dataset.cvdisaster import CVDisasterDatasetEval
from sample4geo.transforms import get_transforms_val
from sample4geo.evaluate.cvusa_and_cvact import evaluate
from sample4geo.model import TimmModel


@dataclass
class Configuration:
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    # Override model image size
    img_size: int = 384
    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0, 1)
    normalize_features: bool = True
    split_key: str = "val60"  # 与本轮训练共用随机 6:4 测试集及候选库。
    # Dataset
    data_folder = "/scratch/liuzeyu/heyinxi/dataset/m1749324/CVIAN"
    # Checkpoint to start from
    checkpoint_start = None  # 必须从命令行指定本轮最佳权重，禁止默认评估初始 CVUSA 权重。
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------#
#  Config                                                                      #
# -----------------------------------------------------------------------------#


config = Configuration()


if __name__ == '__main__':

    # 独立复评必须明确输入 checkpoint；训练脚本末尾也会自动复评本轮最佳权重。
    parser = argparse.ArgumentParser(description='随机 6:4 Geo 最佳权重复评')
    parser.add_argument('--checkpoint', required=True, help='本轮最佳 .pth 权重路径')
    args = parser.parse_args()
    config.checkpoint_start = args.checkpoint

    # -----------------------------------------------------------------------------#
    #  Model                                                                       #
    # -----------------------------------------------------------------------------#
    print("\nModel: {}".format(config.model))

    model = TimmModel(config.model,
                      pretrained=True,
                      img_size=config.img_size)
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    image_size_sat = (img_size, img_size)
    new_width = config.img_size * 2
    new_hight = round((224 / 1232) * new_width)
    img_size_ground = (new_hight, new_width)
    # load pretrained Checkpoint
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
    # Model to device
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # -----------------------------------------------------------------------------#
    #  DataLoader                                                                  #
    # -----------------------------------------------------------------------------#
    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   )
    # Reference Satellite Images
    reference_dataset_test = CVDisasterDatasetEval(data_folder=config.data_folder,
                                                   split="test",
                                                   img_type="reference",
                                                   transforms=sat_transforms_val,
                                                   split_key=config.split_key
                                                   )
    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    # Query Ground Images Test
    query_dataset_test = CVDisasterDatasetEval(data_folder=config.data_folder,
                                               split="test",
                                               img_type="query",
                                               transforms=ground_transforms_val,
                                               split_key=config.split_key
                                               )
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))

    # -----------------------------------------------------------------------------#
    #  Evaluate                                                                    #
    # -----------------------------------------------------------------------------#
    print("\n{}[{}]{}".format(30 * "-", "CVDISASTER", 30 * "-"))

    r1_test = evaluate(config=config,
                       model=model,
                       reference_dataloader=reference_dataloader_test,
                       query_dataloader=query_dataloader_test,
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)
