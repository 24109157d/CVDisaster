# CVDisaster 环境与代码同步说明

## 文档定位与当前状态

本文件记录两条原作者实验线的环境准备步骤和检查点。当前只依据仓库代码、原 README 和武汉大学集群使用文档整理依赖，尚未在目标服务器安装或验证。`requirements.txt`、`requirements-est.txt` 是环境清单，不是精确锁定的可运行环境快照；服务器的 Python、GPU 驱动、CUDA、可用 Conda 路径和 Slurm 账号仍待核实。验证后再记录确切版本与运行结果。

本地只修改代码、文档并管理 Git 版本。数据集、预训练权重、Conda 环境和训练产物放在服务器。两个实验线分别使用 Conda 环境，以免 PyTorch 和 TensorFlow 的依赖互相影响。密码、访问令牌及其他凭据不得写入仓库或命令示例。

## 一、文件与依赖关系

| 实验线 | 服务器工作目录（相对仓库根目录） | 根目录依赖清单 | 代码入口 |
| --- | --- | --- | --- |
| 作者原实验：跨视角定位 Geoloc | `Author CVDisaster/geolocalization/` | `requirements.txt` | `0_prepare_splits_and_data.py`、`1_train_cvdisaster.py`、`2_eval_cvdisaster.py` |
| 作者原实验：灾损分类 Est | `Author CVDisaster/disaster_perception_mapping/` | `requirements-est.txt` | `1_svi.py`、`1_sat.py`、`2_cv.py`、`3_estimation.py` |
| 自己的复现与改进 | `ME CVDisaster/` | 按实验记录 | `scripts/reproduction/`、`scripts/experiments/` |

作者原有的 `Author CVDisaster/geolocalization/requirement.txt` 保留作来源参考；本项目后续以根目录 `requirements.txt` 为定位实验的安装入口。新增依赖来自代码导入：定位预处理使用 `geopandas` 和 Pillow，图像加载及变换使用 OpenCV；分类辅助脚本使用 `geojson`、`rasterio`、`filetype`。作者 README 为分类实验使用 TensorFlow 2.10.1 GPU 容器，当前清单记录该版本基线，但未验证改用 Conda 环境后的 GPU 兼容性。

## 二、本地开发与 GitHub 同步

本地仓库的 `origin` 是自己的 GitHub 仓库，`upstream` 是作者仓库。每次修改先检查变更，审阅后再提交并推送；服务器只拉取已经推送的代码。以下命令在各自的仓库根目录执行：

```bash
git status --short
git diff --check
git add <本次需要提交的代码和文档>
git commit -m "<本次改动说明>"
git push origin main
```

服务器获取代码时，先进入本项目的实际检出目录，确认工作区无未处理变更，再执行：

```bash
git status --short
git pull --ff-only origin main
git rev-parse HEAD
```

服务器项目绝对路径尚未确认；不得沿用其他项目的目录。数据集、权重及训练输出不通过 GitHub 传输。根目录 `.gitignore` 已排除两条实验线的数据目录、权重和主要训练产物；加入新类型的大文件前仍须检查 `git status`，必要时补充规则。忽略规则只作用于尚未纳入 Git 的文件。

## 三、服务器环境创建与安装

先在服务器确认 `conda --version`、`python --version`、`nvidia-smi`、可用磁盘空间，以及集群对登录节点和软件安装的规定。下面以 Python 3.10 作为待验证的创建基线，环境名称仅用于区分两条实验线；若服务器 CUDA/驱动或软件策略不匹配，先调整清单与文档，再安装。所有命令在服务器执行。

### 3.1 定位实验

从仓库根目录执行：

```bash
conda create -n cvdisaster_geoloc python=3.10
conda activate cvdisaster_geoloc
python -m pip install -r requirements.txt
python -c "import torch, timm, albumentations, cv2, geopandas, transformers; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` 应在所申请的 GPU 计算节点上检查；登录节点上显示 `False` 不能单独说明训练环境不可用。PyTorch 安装包与目标 CUDA 的匹配方式待服务器检查后确定。本步骤不在本地安装这些依赖。

### 3.2 灾损分类实验

从仓库根目录执行：

```bash
conda create -n cvdisaster_est python=3.10
conda activate cvdisaster_est
python -m pip install -r requirements-est.txt
python -c "import tensorflow as tf, gcvit, tensorflow_addons, sklearn, geojson, rasterio, filetype; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

作者使用的是 `tensorflow/tensorflow:2.10.1-gpu` 容器，这不等于上述 Conda 安装已经具备相同的 CUDA/cuDNN 运行环境。应先核实集群支持的 GPU 软件栈，再确认安装方式及版本；GPU 可见性也应在计算节点检查。不要在登录节点运行训练或大规模预处理。

## 四、数据、权重与工作目录检查

从 [作者 README](README.md) 指向的 CVIAN 发布页获取数据集，并在服务器分别放到以下位置。两条实验线的脚本均以当前工作目录解析 `./CVIAN`，所以运行前先进入对应子目录。

```text
dataset/raw_dataset/CVIAN/
├── 00_SVI/
├── 01_Satellite/
└── 02_Position/CVIAN_position.geojson

dataset/raw_dataset/CVIAN/
├── 00_SVI/
├── 01_Satellite/
└── 02_Position/
```

定位预处理会在 `dataset/raw_dataset/CVIAN/` 生成对齐图像和划分 CSV。定位训练脚本默认读取当前目录下的 `weights_e40_98.6830.pth`，作者 README 指向 Sample4Geo 的 CVUSA 预训练权重；训练前须确认下载到的实际文件与 `checkpoint_start` 一致。定位评估脚本也要把 `checkpoint_start` 改成实际训练权重路径，并核对所选 `split_key`。分类脚本通过 `pretrain=True` 请求 GCViT 预训练参数，单视角和双视角训练分别生成 `bestSVI.h5`、`bestSAT.h5`、`<训练比例>_bestCV.h5`；`3_estimation.py` 默认读取 `0.5_bestCV.h5`。这些文件均不得提交到 Git。

执行前的最小文件检查示例（先进入相应子目录）：

```bash
# 在 Author CVDisaster/geolocalization/
test -f CVIAN/02_Position/CVIAN_position.geojson
test -d CVIAN/00_SVI
test -d CVIAN/01_Satellite
test -f weights_e40_98.6830.pth

# 在 Author CVDisaster/disaster_perception_mapping/
test -d CVIAN/00_SVI
test -d CVIAN/01_Satellite
```

这些检查只验证文件存在，不代表图像配对、数据完整性或模型权重正确。正式训练命令、具体 Slurm 资源申请、数据校验和实验指标将在服务器条件与数据核实后另行记录。武汉大学超算的通用 Slurm 流程见 [项目操作手册.md](项目操作手册.md)。

## 五、环境验证记录

服务器环境确认后，为每条实验线记录：Git 提交号、Conda 环境名与 Python 版本、`python -m pip freeze` 输出、GPU/驱动/CUDA 信息、数据集来源与校验结果、预训练权重来源、导入检查结果，以及首次小规模运行的日志位置。记录中不得包含凭据。只有完成对应计算节点的导入和 GPU 检查，才能将该环境标记为“已验证”。



