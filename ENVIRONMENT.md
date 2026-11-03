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

## 九、恢复完整训练（2026-11-02）

用户告知已充值，并明确改为直接完整训练、暂不执行 smoke。只读检查确认完整任务 `3780573`、`3780728` 仍为挂起状态且运行时间为零；Geoloc 的正式划分 CSV 尚不存在，预训练文件和原始图像目录仍在。

已执行 `scancel 3802343` 取消尚未运行的 Geoloc smoke，并执行 `scontrol release 3780573 3780728` 恢复原完整任务。没有重新提交训练副本、重装依赖或修改训练超参数。仍使用原始提交 `4785bbc03c5d604a0b00dc01e2cfdede04eae0b0` 的两份完整训练脚本：Geoloc 10 轮，Est 200 轮后推理，每任务 1 GPU、8 CPU、60000 MB 主机内存、最长 7 天。解除挂起后即时状态为 PENDING，需以之后的 Slurm 状态和启动日志确认实际运行，不能据充值或解除挂起推断训练成功。

本次进一步核对发现，两个完整任务重新评估后均为 `PENDING (AssocGrpCpuLimit)`，开始时间 `N/A`，运行时间零，尚未生成训练日志。`sacctmgr` 显示父账号 `wanyi` 的 `GrpCPUs=0`、`GrpTRES=cpu=0,gres/gpu=16`；个人 `liuzeyu` 在 `a100x4` 的限额为 64 CPU / 4 GPU / 2 节点，父账号限额仍阻止调度。当次 `squeue -A wanyi -t R` 没有运行任务，因此应请超算管理员确认充值及父账号 CPU 调度配额恢复情况。未修改账号、配额或 QOS，也不需要为此更改依赖、重复提交或减少 batch。

## 十、取消旧任务并重新提交（2026-11-02）

用户明确要求取消 `3780573`、`3780728` 并各重新提交一次。执行前发现旧 Geoloc 已在 `g0165` 运行，取消时累计 `00:31:05`；日志含 `GEOLOC_GPU_OK 1.13.1+cu116 NVIDIA A100-SXM4-40GB`，说明其 PyTorch 实际 GPU 运算已通过。当时仍在生成 `00_SVI_aligned`，尚未生成正式 `train_subarea5.csv`、`val_subarea5.csv`，未进入训练阶段。旧 Est 尚未训练。已确认两条旧任务取消并退出队列。

按同一份已发布脚本、相同资源配置和源代码提交 `4785bbc03c5d604a0b00dc01e2cfdede04eae0b0` 各提交一次：Est 新 JobID `3952478`，Geoloc 新 JobID `3952479`。提交时两者均为 `PENDING (Priority)`，后续状态以 Slurm 为准，不等同于训练通过。没有重新安装依赖、修改作者模型或调整超参数。

新 Geoloc 将重新完成被中断的对齐预处理；只更新作者脚本生成的派生对齐图像及划分，原始 `00_SVI`、`01_Satellite` 和预训练权重不变。运行副本和正式训练输出使用各自的新 JobID 目录，不覆盖旧任务输出。新日志为 `logs/est-3952478.log`、`logs/geoloc-3952479.log`；提交记录为 `runs/resubmit-full-20261102-est.txt`、`runs/resubmit-full-20261102-geoloc.txt`。此前父账号零 CPU 配额是之前查询时的状态；本次已观察到实际 GPU 分配，不能继续将该旧限额快照作为当前结论。

## 十一、Slurm 用户环境获取失败与提交方式修正（2026-11-02）

任务 `3952478`、`3952479` 随后均变为 `user_env_retrieval_failed_requeued_held`，`Restarts=1`、运行时间零，训练日志没有生成。按 [Slurm sbatch 官方说明](https://slurm.schedmd.com/sbatch.html)，`--export=ALL,变量=值` 这种显式变量列表会隐式触发 `--get-user-env`；该机制获取用户登录环境失败或超时时会重新排队并挂起。之前的提交命令使用了这种写法，应该修正；这不是 Python 模型报错的证据。登录初始化中存在 Conda hook，但此次没有取得 slurmd/slurmstepd 的管理员日志，无法断定底层失败具体来自超时、命名空间或其他节点问题。

已取消这两个失败挂起的任务，在提交 shell 内显式导出代码提交号、清除该进程中的用户环境获取选项，再使用纯 `--export=ALL` 各重新提交一次。没有改动用户全局 shell 配置、作者代码、模型参数或依赖。新 Est 为 `3952585`，新 Geoloc 为 `3952586`，提交时均为 `PENDING (Priority)`，实际启动情况仍须检查。新的提交记录为 `runs/full-envfix-20261102-est.txt`、`runs/full-envfix-20261102-geoloc.txt`。

```bash
export CVDISASTER_COMMIT=4785bbc03c5d604a0b00dc01e2cfdede04eae0b0
unset SBATCH_GET_USER_ENV SLURM_GET_USER_ENV SBATCH_EXPORT
sbatch --export=ALL /scratch/liuzeyu/heyinxi/CVDisaster/slurm/train_est.sbatch
sbatch --export=ALL /scratch/liuzeyu/heyinxi/CVDisaster/slurm/train_geoloc.sbatch
```

后续传递提交号采用上面的写法；不再使用逗号附加显式变量列表。当时的 Smoke 方案示例也同步修正；该方案及测试脚本现已按用户要求移除。

## 十二、完整训练执行结果：两项失败（2026-11-02）

22:04 的 Slurm 查询确认，两项任务均已结束，状态 `FAILED`、退出码 `1:0`。没有完成任何训练 epoch，也没有完成一次训练参数更新，没有生成新的训练模型权重。

| 模块 | JobID | 开始 / 结束（北京时间） | 总耗时 | 失败阶段 |
| --- | --- | --- | --- | --- |
| Est | 3952585 | 16:12:05 / 16:53:54 | 00:41:49 | 首个 GCViT 分支的预训练权重加载 |
| Geoloc | 3952586 | 16:16:36 / 17:39:11 | 01:22:35 | 第 1 轮第 1 个 batch 的前向传播 |

### 已验证与失败原因

- 两条任务都通过实际 GPU 矩阵计算：`EST_GPU_OK 2.10.1 ... GPU:0`、`GEOLOC_GPU_OK 1.13.1+cu116 NVIDIA A100-SXM4-40GB`。此前 Slurm 用户环境获取问题已越过。
- Est 已识别 A100，加载 cuDNN 8100，进入双分支模型构建；随后 GCViT 调用 Keras `get_file`，试图下载 `gcvit_tiny_weights.h5`，计算节点无法解析外网域名，报 `socket.gaierror: [Errno -2] Name or service not known` 并退出。实际预训练文件仍保存在 `pretrained/keras/datasets/`。本机缓存的 Keras 2.10.0 源码确认：`get_file(cache_dir=None)` 默认使用 `~/.keras`（不可写时使用 `/tmp/.keras`），并不读取此处设置的 `KERAS_HOME`；当次默认 home datasets 目录没有对应文件。这是本次缓存映射配置未对准实际加载器的问题，不能认定为权重丢失，也不能将可选 TensorRT/ptxas 提示当作最终异常原因。没有进入 `model.fit`，没有生成 `0.5_bestCV.h5` 或 `estimation.txt`。
- Geoloc 完成图像对齐、划分、ConvNeXt Base/CVUSA 权重加载、GPU 特征提取、训练前 zero-shot 评估及相似性采样。在第 1 轮首个 batch 前向传播时，单张 A100 40GB 无法容纳作者默认 batch 128：已分配 37.54 GiB，只余 215.31 MiB，再申请 576 MiB 导致 `torch.cuda.OutOfMemoryError`。已保留的显存与实际已分配显存接近，日志没有证据支持仅靠处理显存碎片即可解决。未完成第一个训练 batch，不能把模型初始化或 zero-shot 评估称为微调成功。

### 本次实际生成的文件与初步指标

- 原始数据根目录：`/scratch/liuzeyu/heyinxi/dataset/m1749324/CVIAN/`。生成 `00_SVI_aligned/` 内 4121 张派生对齐图像和根目录 23 个 CSV 划分文件；原始图像目录作为输入使用。
- 当前 subarea5 CSV：训练 3977 行、验证 268 行。只读核查得到训练唯一 ID 3854 个、验证唯一 ID 267 个，分别有 123 行和 1 行重复 ID，两个集合的 ID 无交叉。后续需核查作者空间连接生成重复行的原因，不能将 CSV 行数直接当作不同图像数量。
- Geoloc 训练前验证集 zero-shot 结果：Recall@1 35.8209%、Recall@5 51.1194%、Recall@10 60.8209%、作者日志的 top1% 项 41.4179%。这些来自加载的 CVUSA 预训练权重，未进行本次训练更新，且基于上述含重复行的当前 CSV，不能作为正式微调结果或直接对标论文。
- 完整 Slurm 日志：`/scratch/liuzeyu/heyinxi/logs/est-3952585.log`、`/scratch/liuzeyu/heyinxi/logs/geoloc-3952586.log`。本机副本在 `.local-cache/run-logs/20261102/`，被 Git 忽略。
- Geoloc 内部记录目录：`/scratch/liuzeyu/heyinxi/runs/geoloc-3952586/Author CVDisaster/geolocalization/cvdisaster/convnext_base.fb_in22k_ft_in1k_384/172033/`，只有 `train.py` 配置快照和 `log.txt`，无训练 checkpoint。
- Est 工作副本：`/scratch/liuzeyu/heyinxi/runs/est-3952585/Author CVDisaster/disaster_perception_mapping/`，没有训练权重或推理结果。

本次结果查询没有修改作者代码、batch、优化器、依赖、预训练权重或重新提交任务。后续需先修正 Est 缓存映射，并确认 Geoloc 单卡显存及数据重复问题的处理方案，再进行验证和重跑。

## 十三、修复 Est 默认权重缓存（2026-11-02，离线验证通过）

按用户要求保持本地修改 → 推送 GitHub → 超算拉取 → 执行的流程。新增 `scripts/prepare_est_cache.py`：先核对 scratch 中现有 GCViT 权重的完整 SHA256，再创建 `~/.keras/datasets/gcvit_tiny_weights.h5` 指向该文件的精确软链接。只创建缺失链接，遇到已有普通文件、错误链接或不可写默认缓存目录即报错，不覆盖任何已有内容，不改 HOME 或全局 shell 初始化。

`slurm/train_est.sbatch` 增加只读缓存检查，避免仅设置 `KERAS_HOME` 却漏检真实加载路径。作者模型、训练参数和依赖保持原样。下次运行新版 Est sbatch 时，`CVDISASTER_COMMIT` 必须包含新增的缓存检查脚本，不能继续把旧的 `4785bbc` 与新版 sbatch 混用。

当时使用的验证程序（位于历史提交 `fa8faa6`，验证结束后已清理）：使用现有 Est 环境，验证真实 `tf.keras.utils.get_file` 返回已缓存文件；拦截下载入口，任何下载尝试都会失败；随后以 224×224 CPU 输入初始化 GCViT Tiny、加载已有预训练权重并执行一次前向，检查文件 SHA256 未变。该验证不训练、不改正式输入尺寸、不安装依赖、不申请 GPU，也不代替完整训练的显存及反向传播验证。实际执行结果见下文。

修复代码提交 `fa8faa684adce849242ac2456947eaafda7f2a98` 已先推送 GitHub，再在超算拉取并核对提交号，随后执行修复脚本。实际创建的链接为 `/home/liuzeyu/.keras/datasets/gcvit_tiny_weights.h5` → `/scratch/liuzeyu/heyinxi/pretrained/keras/datasets/gcvit_tiny_weights.h5`，没有复制或重新下载权重。

免费 CPU 验证 JobID `3954386`，节点 `n0358`；主作业和 batch 均为 `COMPLETED`、退出码 `0:0`，耗时 `00:22:46`。结果 JSON 为 `status=passed`，真实 Keras 默认缓存命中，`download_attempts=0`，GCViT Tiny 以 `pretrain=True` 成功加载现有 H5 权重，CPU 前向输出为 `[1,1000]`、有限且概率和为 1，验证后原始权重 SHA256 保持一致。日志标记：`KERAS_GET_FILE_CACHE_HIT`、`EST_PRETRAINED_OFFLINE_LOAD_VERIFIED`。

验证日志为 `/scratch/liuzeyu/heyinxi/logs/verify-est-cache-3954386.log`，结果为 `/scratch/liuzeyu/heyinxi/est-install/cache-verify-3954386/result.json`；本机副本放在 `.local-cache/run-logs/20261102/`，不上传 GitHub。此次只修复和验证预训练加载，没有重新安装依赖，也没有重新提交 Est 完整训练。后续完整训练仍需验证原始分辨率、batch=12 下的显存、反向传播和持续运行，不能把缓存验证通过视为 200 轮训练已通过。

## 十四、清理临时验证及 Smoke 代码（2026-11-02）

用户确认不再进行小实验，已移除 `SMOKE_TEST_PLAN.md`、`scripts/smoke/` 三个 Python 文件、`slurm/smoke.sbatch`、`scripts/verify_est_cache.py`、`slurm/verify_est_cache.sbatch`，合计七个文件。历史任务、验证结论和日志路径保留，程序可从 Git 历史恢复，不再作为当前运行入口。

保留 `scripts/prepare_est_cache.py`：它是正式 `slurm/train_est.sbatch` 归档和调用的缓存修复/预检查依赖，不属于可删除的临时验证程序。已有默认 Keras 缓存软链接、scratch 预训练权重、Conda 环境、作者模型及两个完整训练入口均保持不变。清理过程不运行训练、不安装依赖、不删除历史训练或验证结果。

## 十五、Geoloc 四卡训练与已有数据复用（2026-11-03）

按用户最终确认，Geoloc 调整为单节点四张 A100，训练全局 batch=64（每卡约 16 对图片）、基础学习率 `5e-4`。评估全局 batch=128，同样使用四卡 DataParallel（完整 batch 每卡约 32 张图片）；初始评估、每轮评估和难负样本特征提取均保留作者的多卡路径。训练仍使用 DataParallel 汇总全局特征计算损失，10 轮、混合精度、图像尺寸及现有数据划分不变。最初的单卡评估设置已按用户后续要求撤回，四张卡在作业期间持续分配。

学习率 warm-up/cosine 调度改在首次自定义采样后初始化，以采样后的 DataLoader 长度计算步数。此处没有修改重复 ID、采样策略或空间划分方式。

新增正式运行辅助脚本 `scripts/prepare_geoloc_data.py`，由 Geoloc sbatch 随同训练源码归档并调用。两份 subarea5 CSV 和对齐目录已存在时，检查两列路径、图片文件存在且非空、图片 ID/类别一致、训练验证 ID 无交叉，然后直接复用；重复行保留。若已有部分产物或检查失败，明确报错且不覆盖文件。仅当两份 CSV 和对齐目录均不存在时才执行作者预处理。该检查不解码全部图像，也不代表训练或四卡显存验证通过。

本次仅修改并检查脚本，不提交 GPU 作业、不重新安装依赖、不覆盖原始数据、预训练权重或旧训练输出。上次 Geoloc 没有完成参数更新，新训练仍从已有 CVUSA 预训练权重开始；已完成的预处理数据可复用。

## 十六、Geoloc 改为两卡以降低排队资源要求（2026-11-03）

四卡完整训练任务 `3956471` 已使用提交 `ce1420d` 提交；本次检查时仍为 `PENDING (Resources)`，尚未开始运行。用户确认在持续等待资源时改用两张 GPU。当前配置改为单节点两张 A100、`gpu_ids=(0,1)`，提交脚本逐卡运算检查也改为两卡。

全局训练 batch=64、基础学习率 `5e-4`、全局评估 batch=128 保持不变：训练每卡约 32 对图片，评估完整 batch 每卡约 64 张图片。DataParallel 仍汇总全局特征计算对比损失，GPU 数变化没有改变全局 batch，因此本次不再次缩放学习率。保留 10 轮、混合精度、8 核 CPU、60000M 内存及 7 天时限；复用已经检查通过的预处理数据，重复 ID 保持原样，不重新安装依赖。

替换提交时先同步 GitHub 并验证脚本，再确认旧任务仍处于资源等待；仅取消尚未运行的旧任务后提交一次新版。若旧任务已启动，保留该运行任务并停止替换，避免重复训练。实际新 JobID、源代码提交号、日志和运行目录记录在超算 `runs/submit-geoloc-<JobID>.json` 中。两卡的实际峰值显存仍需完整训练 batch 验证。
