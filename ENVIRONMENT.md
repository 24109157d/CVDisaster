# CVDisaster 环境与代码同步说明

## 文档定位与当前状态

本文件记录两条原作者实验线的环境准备步骤和检查点。定位环境于 2026-10-25 通过免费 `hpxg` 分区完成依赖补装与 CPU 项目验证，结果记录在第五节。根目录 `requirements.txt` 已固定本次验证的定位依赖版本，完整 53 包版本快照见 `requirements-geoloc.lock.txt`；分类独立环境 `cvdisaster_est` 也已完成安装、依赖检查和 CPU 项目验证，结果见第六节，直接依赖入口为 `requirements-est.txt`。GPU 驱动、GPU 运行和训练结果尚未验证。

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

服务器项目路径为 `/scratch/liuzeyu/heyinxi/CVDisaster`。数据集、权重及训练输出不通过 GitHub 传输。根目录 `.gitignore` 已排除两条实验线的数据目录、权重和主要训练产物；加入新类型的大文件前仍须检查 `git status`，必要时补充规则。忽略规则只作用于尚未纳入 Git 的文件。

## 三、服务器环境创建与安装

定位环境的 Python 3.10.22 和依赖 CPU 运行已核实；分类环境使用 Python 3.10.22，已通过 CPU 项目验证。新建环境前应确认集群调度规则、存储空间和 GPU 软件栈；若需要改变依赖版本，先与用户确认。以下命令在服务器执行。

### 3.1 定位实验

现有 `cvdisaster_geoloc` 环境及 PyTorch 已安装，不要重复创建环境或重装 PyTorch。本次剩余依赖通过 `/scratch/liuzeyu/heyinxi/torch-install/install_remaining.sbatch` 离线安装，具体版本和日志见第五节。需补装时先核对当前环境和任务状态，避免重复提交安装任务。

```bash
conda activate cvdisaster_geoloc
cd /scratch/liuzeyu/heyinxi/torch-install
squeue -u "$USER"
# 导入和计算验证通过 Slurm 运行，不在登录节点直接执行。
```

`torch.cuda.is_available()` 应在所申请的 GPU 计算节点上检查；登录节点上显示 `False` 不能单独说明训练环境不可用。现有 PyTorch 是 CUDA 11.6 构建，本次补装保持原样；GPU 驱动匹配和实际 GPU 运行仍待验证。本步骤不在本地安装这些依赖。

### 3.2 灾损分类实验

独立环境 `cvdisaster_est` 已完成安装和 CPU 项目验证，具体版本、日志及验证范围见第六节。无需重复创建环境或重装依赖。

```bash
conda activate cvdisaster_est
cd /scratch/liuzeyu/heyinxi/CVDisaster
# 运行前先通过 GitHub 同步代码，导入和计算检查通过 Slurm 执行。
```

作者使用 `tensorflow/tensorflow:2.10.1-gpu` 容器。改用 Conda 后，本次为 Est 环境单独安装 CUDA 11.2.2、cuDNN 8.1.0.77，并配置环境内的动态库路径；定位环境保持原样。`requirements-est.txt` 是直接依赖入口，`constraints-est.txt` 固定兼容版本。实际 GPU 运行仍须在 GPU 计算节点验证。

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

### 2026-10-25 定位依赖 CPU 验证通过（GPU 未验证）

- 服务器仓库：`/scratch/liuzeyu/heyinxi/CVDisaster`。
- Conda 环境：`/scratch/liuzeyu/anaconda3/envs/cvdisaster_geoloc`；Python 实测为 `3.10.22`。测试代码提交号为 `9d6b5478b68b73600f453633d7eaaa0fb64a07a6`。
- 保留原有 `torch==1.13.1+cu116`、`torchvision==0.14.1+cu116`、`timm==0.8.19.dev0`，安装计划不包含这三个包。
- SciPy、Albumentations、scikit-learn、Transformers 采用作者要求中的最低版本，分别为 `1.10.0`、`1.3.0`、`1.2.1`、`4.27.0`；NumPy 固定为 `1.26.4`。
- 官方 PyPI 在本机可达，在登录节点连接超时。本机从官方 PyPI 下载 47 个 Linux / CPython 3.10 wheel 并生成 SHA256，后追加 `hf-xet==1.5.0`，合计 48 个；超算从阿里云镜像下载相同版本，所有文件与官方包的 SHA256 一致。未修改全局 pip 源配置。
- 安装目录：`/scratch/liuzeyu/heyinxi/torch-install/`；wheel 在 `remaining-wheels/`，哈希清单为 `remaining-wheels/SHA256SUMS`。
- 安装脚本：`install_remaining.sbatch`、`install_remaining.py`；检查脚本：`verify_remaining.py`。脚本先离线解析，再拒绝任何涉及 torch、torchvision、timm 的变更，仅对剩余计划执行 `--no-index --no-deps` 安装。
- 第一次任务 `3766020` 因 SHA256 清单的 Windows 换行在校验阶段退出，未安装任何包；修正换行后提交 `3766022`，节点 `n0334`，免费 `hpxg`，2 CPU、8 GiB 内存。
- `3766022` 成功安装 40 个缺失包，并通过 `pip check`。项目导入检查发现 `/home/liuzeyu/.local/lib/python3.10/site-packages/transformers` 的不完整残留遮住环境内完整的 Transformers，任务因验证失败而退出；这不是已安装版本之间的依赖冲突。
- 已通过 `conda env config vars set --prefix /scratch/liuzeyu/anaconda3/envs/cvdisaster_geoloc PYTHONNOUSERSITE=1` 为本环境隔离用户目录。当前已激活的终端需要 `conda deactivate` 后重新 `conda activate cvdisaster_geoloc` 才能应用设置。批处理脚本显式 `export PYTHONNOUSERSITE=1`。
- 用户目录原先提供的 `hf-xet==1.5.0` 已以同一版本补入环境。最终任务 `3766277` 在 `n0334` 完成，状态 `COMPLETED`、退出码 `0:0`、耗时 `00:17:57`；未重装前述 40 个包或 PyTorch 栈。
- 日志：`install-remaining-3766022.log`、`install-remaining-3766277.log`；离线解析结果：`resolve-offline.json`。
- 已通过：隔离环境中的 `pip check`（`No broken requirements found.`）、依赖导入与实际路径、NumPy/PyTorch 转换、Safetensors 张量读写、torchvision CPU NMS、地理坐标转换、GeoJSON 读取、空间连接、项目模块导入、训练/验证图像变换、无预训练权重下载的 ConvNeXt CPU 前向计算。最终标记为 `GEOLOC_DEPENDENCIES_VERIFIED_CPU`。GPU 与实际训练尚未验证。
- 原始快照：安装目录中的 `requirements-geoloc-freeze.txt`。该文件含本地 wheel 路径；仓库 `requirements-geoloc.lock.txt` 将其转换为精确版本号，记录 53 个包，用于复现本次环境。主入口 `requirements.txt` 列出定位直接依赖与已保护的 PyTorch 栈。安装目录另保存完整版本快照；依赖清单与环境记录按 GitHub 同步流程提交。

服务器环境确认后，为每条实验线记录：Git 提交号、Conda 环境名与 Python 版本、`python -m pip freeze` 输出、GPU/驱动/CUDA 信息、数据集来源与校验结果、预训练权重来源、导入检查结果，以及首次小规模运行的日志位置。记录中不得包含凭据。只有完成对应计算节点的导入和 GPU 检查，才能将该环境标记为“已验证”。

## 六、Est 独立 Conda 环境配置记录（CPU 验证通过，2026-10-25）

不用 Docker 本身不要求修改模型代码。当前保留作者的 TensorFlow 2.10.1，使用独立环境 `/scratch/liuzeyu/anaconda3/envs/cvdisaster_est`，与定位环境隔离。模型代码尚未因本次安装修改；实际运行仍需数据集、预训练权重和合适的 Slurm 资源。

- 原始依据：作者 README 的 `tensorflow/tensorflow:2.10.1-gpu` 镜像；[TensorFlow 构建兼容表](https://www.tensorflow.org/install/source#gpu) 的 2.10 系列使用 Python 3.7–3.10、CUDA 11.2、cuDNN 8.1；[Addons 兼容表](https://github.com/tensorflow/addons#c-custom-op-compatibility-matrix) 中 0.18.0 对应 TensorFlow 2.10。
- Python 3.10.22、TensorFlow 2.10.1、Keras 2.10.0、GCViT 1.1.6、Addons 0.18.0、NumPy 1.23.5；其余历史接口约束见 `constraints-est.txt`。作者未锁定的间接依赖按该基线选择，未升级作者指定的 TensorFlow。
- 安装工作目录：`/scratch/liuzeyu/heyinxi/est-install/`。Python 包使用官方 PyPI 下载的 104 个 Linux wheel 生成 SHA256，超算从镜像下载并核对同一哈希。
- GPU 运行库从 conda-forge 官方源下载：`cudatoolkit-11.2.2-hc23eb0c_13.conda`（官方 SHA256 核对），`cudnn-8.1.0.77-h90431f1_0.tar.bz2`（官方历史包仅提供 MD5，已核对）。已安装到 Est 环境，未改定位环境。
- 首次创建任务 `3767461` 因缓存中旧版 Conda wheel 是 Python 3.9 构建而解析失败。新任务 `3767475` 复用定位环境的 32 个 Conda 基础运行包显式创建 Python 3.10 环境，状态 `COMPLETED`、退出码 `0:0`、耗时 `00:30:50`；没有复制其 pip 包或 PyTorch。
- `create_est.sbatch` 创建环境并设置 `PYTHONNOUSERSITE=1`。`install_est.sbatch` 离线安装 CUDA/cuDNN、配置本环境库路径、安装 pip 依赖并验证，使用免费 `hpxg`；安装和 CPU 测试均在该分区完成。
- 安装及验证任务 `3767479`，状态 `COMPLETED`、退出码 `0:0`、耗时 `02:21:26`，节点 `n0358`，2 CPU、8 GiB 内存。CUDA/cuDNN 已安装，104 个 wheel 哈希复核通过，离线解析成功，102 个 Python 包已安装成功，`pip check` 已输出 `No broken requirements found.`，CPU 运行验证已通过。日志为安装目录中的 `install-est-3767479.log`；本轮未下载或安装 torch、torchvision、timm。
- 已通过：隔离环境中的 `pip check`；TensorFlow/Addons/GCViT、NumPy、scikit-learn、Pillow、GeoJSON、Rasterio、filetype、h5py 导入和实际路径；TensorFlow CUDA 构建版本；CUDA 动态库加载与 cuDNN 版本 `8100`；Addons GELU 和 TensorFlow 矩阵计算；PNG 解码、GeoJSON/GeoTIFF 读写；原代码 WarmUpCosine；双分支 GCViT CPU 前向和 H5 权重保存/加载。模型测试使用合成的 224×224 输入、无预训练下载，没有启动原脚本训练。模型验证标记为 `EST_COUPLED_GCVIT_FORWARD_H5_OK`，最终标记为 `EST_DEPENDENCIES_VERIFIED_CPU`。
- 该验证在无 GPU 的 `n0358` 上执行，缺少 NVIDIA 驱动的提示符合 CPU 节点条件。TensorRT 是可选功能，当前未安装，也没有测试 TF-TRT。实际 GPU 运行、原始大分辨率输入及真实训练仍待计算节点实验验证。
- 版本快照：安装目录保存 `requirements-est-freeze.txt`；仓库 `requirements-est.lock.txt` 固定 104 个 Python 包的精确版本，去除了本地 wheel 路径。CUDA/cuDNN 由 Conda 管理，单独记录于上文。测试所用源代码提交为 `9d6b5478b68b73600f453633d7eaaa0fb64a07a6`。
- 本次代码和依赖清单仍按本地修改 → GitHub → 超算拉取同步；安装辅助文件放在安装目录，不直接覆盖超算仓库源码。

## 七、完整 GPU 训练任务（2026-10-26 已提交，等待调度）

用户指定两条实验线均直接完整训练，固定资源为 `a100x4`、`wanyi`、单节点、单任务、8 CPU、1 GPU。脚本为 `slurm/train_geoloc.sbatch` 和 `slurm/train_est.sbatch`，额外设置内存 60000 MB（分区每 GPU 上限）和最长 7 天。未调整作者模型代码、批量大小、学习率或训练轮数。

- Geoloc：ConvNeXt Base、`subarea5`、batch 128、10 轮，执行作者预处理后训练，每轮评估。原始数据已存在；首次检查未发现对齐图像、划分 CSV 或预训练权重。脚本明确要求划分不存在，再运行作者预处理；重复实验前需确认是否保留已有划分并调整该步骤，禁止直接重复预处理覆盖输入。
- Est：`2_cv.py --tr-ratio 0.5 --backbone tiny`，batch 12、200 轮，训练末尾验证；成功后执行 `3_estimation.py --backbone tiny` 生成逐图推理结果。不包含额外的 SVI/SAT 单视角对照实验。
- 两份脚本均先执行实际 GPU 矩阵计算，失败立即退出，不回退 CPU。使用各自 Conda 环境，不加载案例的 CUDA 12.2。
- 日志：`/scratch/liuzeyu/heyinxi/logs/geoloc-<JobID>.log`、`est-<JobID>.log`。使用提交时指定的 `CVDISASTER_COMMIT` 创建代码运行副本，产物放在 `/scratch/liuzeyu/heyinxi/runs/geoloc-<JobID>/`、`est-<JobID>/`，源码仓库不接收训练输出。
- 预训练文件统一放在 `/scratch/liuzeyu/heyinxi/pretrained/`。Geoloc 的 CVUSA 权重来自作者 README 指向的 [Sample4Geo](https://github.com/Skyy93/Sample4Geo) Google Drive `pretrained.zip`；GCViT Tiny 权重来自安装包指定的 [v1.1.1 发布文件](https://github.com/awsaf49/gcvit-tf/releases/download/v1.1.1/gcvit_tiny_weights.h5)。ConvNeXt 初始化使用 timm 指定的官方 Hugging Face 仓库，固定快照 `2b2698d86523262ca8065d0957c7f80d963bb43f`。计算节点使用已下载缓存。
- 提交代码：`4785bbc03c5d604a0b00dc01e2cfdede04eae0b0`，本地已推送 GitHub，用户在超算拉取后核对一致。未使用 Git bundle。
- 三份预训练文件已到位并校验：CVUSA `weights_e40_98.6830.pth` SHA256 `739cf00d303836fba1195f1bb8d54c4b13c53926ce3212fc9ceced335fc59a23`；ConvNeXt `model.safetensors` SHA256 `bc0cfafc7d7755e8271f51343c4374725bf6409c0434e64028a863c4aca7d9fe`（与官方 Hub LFS 元数据一致）；GCViT `gcvit_tiny_weights.h5` SHA256 `90bf9bfaafd086a831e92edbbfbba9450fd56786c2d39b3ac3f215d0cef54768`。来源和哈希清单保存在超算 `pretrained/pretrained-manifest.json`。
- Est JobID：`3780573`，提交后为 `PENDING (AssocGrpCpuLimit)`；Geoloc JobID：`3780728`，提交后为 `PENDING (Priority)`。提交记录分别在 `runs/est-4785bbc-submission.txt`、`runs/geoloc-4785bbc-submission.txt`。这是正式完整训练任务；此前 `sbatch --test-only` 仅作资源校验，没有启动测试训练。
- 两份脚本本地 Bash 语法检查和 Slurm 资源试算通过。排队期间尚未执行脚本内 GPU 检查、Geoloc 预处理或训练/推理，因此不能宣称 GPU 流程启动成功。Slurm 预计开始时间可能变化，以 `squeue` 为准。

```bash
squeue -j 3780573,3780728
squeue --start -j 3780573,3780728
# 开始运行后查看日志，寻找 EST_GPU_OK、GEOLOC_GPU_OK 和实际训练进度。
tail -n 50 /scratch/liuzeyu/heyinxi/logs/est-3780573.log
tail -n 50 /scratch/liuzeyu/heyinxi/logs/geoloc-3780728.log
```

## 八、本机文件存放与权重清理（2026-10-27）

本项目所有本机文件均保存在 `D:\CVDisaster` 内，后续不得在 D 盘根目录另建本项目的临时目录。此前的 `D:\CVDisaster-geoloc-install`、`D:\CVDisaster-est-install`、`D:\CVDisaster-run-logs` 已移入项目内 `.local-cache/` 对应子目录，并由 `.gitignore` 排除；保留的安装脚本、离线包和历史日志属于安装缓存。

已重新核对超算三份预训练权重，SHA256 与第七节记录完全一致。随后删除本机的 CVUSA 权重、GCViT Tiny 权重、ConvNeXt 初始化权重及 Sample4Geo 压缩包，合计释放 2,444,349,939 字节（约 2.28 GiB）。超算 `/scratch/liuzeyu/heyinxi/pretrained/` 内的文件保持不变，已提交任务继续使用该目录。

权重和数据不上传 GitHub；GitHub 仅保存代码、依赖清单、文件来源、哈希与实验记录。本机中转用于超算无法直接获取的文件，优先在超算下载，传输校验完成后清理中转文件。
