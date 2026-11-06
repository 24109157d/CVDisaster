# CVDisaster 环境与代码同步说明

## 文档定位与当前状态

本文件按时间记录两条实验线的环境准备、代码变更和作业运行。早期“尚未验证”等表述仅代表记录当时的状态；当前配置与最近作业见文末，精简失败原因与教训见 [复现问题复盘.md](复现问题复盘.md)。定位依赖入口为 `requirements.txt`，分类依赖入口为 `requirements-est.txt`，对应完整版本快照见各自的 lock 文件。

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

## 十七、Est 两卡同步训练（2026-11-03）

用户授权 Est 申请两张 A100 并提交完整训练。`2_cv.py --gpus 2` 使用 TensorFlow MirroredStrategy；调度器、SGD 优化器、损失、模型及 compile 均在 strategy.scope 内创建。显式要求两张可见 GPU，开启显存按需分配；默认单卡/TPU 入口保留。全局 batch=12（每卡约 6 对图片）、初始学习率 0.01、warm-up 起点 0.001、200 轮、训练比例 0.5、Tiny 和原始图像尺寸不变。全局 batch 未变，不因卡数变化缩放学习率。

`slurm/train_est.sbatch` 申请两卡，先逐卡执行矩阵计算，再检查分布式归并运算；训练及验证使用两卡。训练后的作者逐图指标计算沿用原代码；独立 `3_estimation.py` 仍使用默认单卡策略执行逐图推理。继续使用现有 cvdisaster_est 环境、scratch 权重和已验证的 Keras 缓存链接，不重新安装依赖或下载权重。实际两卡模型训练的显存和反向传播仍须作业启动后验证。新作业使用独立 JobID 输出目录，提交记录存入超算 runs，不修改正在运行的 Geoloc 作业。

## 十八、按 2025 正式论文配置第二轮代码（2026-11-04）

本节为当前训练入口配置；前文保留历史运行记录。用户已确认本轮修改代码，暂不提交训练作业。依据 `参考论文/2_CVDisaster.pdf` 第 4 节（ISPRS 220，841–854），明确参数以正式论文为准，缺失的实现细节参考作者 GitHub；两任务独立训练、各自划分、各自保存结果。

### 第一轮结果与正确对标

- Geoloc JobID `3956583`：subarea5 空间留出，3854 个训练唯一对、267 个测试唯一对（CSV 分别 3977/268 行）；第 8 轮最佳 R@1/5/10/1% 为 81.7164/95.5224/96.6418/89.5522%。对应新版表 4 Subarea_5 的 80.69/95.15/96.27/89.18%，不能与表 3 的随机比例实验混比。
- Est JobID `3956738`：SGD、两卡、batch 12、200 轮；第 119 轮最佳权重在完整 2061 对上的 OA=70.5000%、macro P=73.0122%、macro R=70.6382%、macro F1=0.71388。对应新版表 6/7 的 5:5，OA=71.18%、P=73.83%、R=71.22%，论文整体 F1=0.73 的汇总定义未明确。

### 第二轮 Geoloc

- 切换为随机 5:5：`split_key=val50`，4121 对按作者排序、配对后 `train_test_split(train_size=0.5, random_state=42)`，预期 2060 对训练、2061 对测试，不分层。
- 两张 GPU、DataParallel、全局训练 batch=64、评估 batch=128。学习率以论文 `1e-4` 随 batch 128→64 减半为 `5e-5`；AdamW、默认 weight decay=0.01、10 轮、1 轮 warm-up 和 cosine 保留。
- 恢复作者在首次自定义采样之前计算步数的位置：2060 对/batch 64 的 DataLoader 为 33 批，预热 33 步、调度总长度 330 步。自定义采样可能舍弃尾部；不再用采样后长度重算调度。
- 模型、CVUSA 权重、朝北对齐、裁剪、缩放、增强、InfoNCE、label smoothing=0.1、混合精度及难负采样沿用作者实现。论文原图尺寸与代码网络输入尺寸分别记录，不额外更改处理链。
- `prepare_geoloc_data.py` 检查实际 CSV 与作者算法的样本及顺序逐项相同、ID 唯一、无训练测试交叉，并在当前 JobID 目录保存两份 CSV 快照及 `geoloc_split_manifest.json`。现有数据残缺直接报错，不覆盖已有划分或历史产物。
- 每轮按测试 R@1 选最佳权重，结束后重新加载最佳权重复评整组指标；独立入口为 `python 2_eval_cvdisaster.py --checkpoint <本轮最佳权重>`，固定同一 val50 测试集，不默认使用初始 CVUSA 权重。
- 对标新版表 3 的 5:5：R@1/5/10/1%=74.48/95.58/98.35/99.37%。

### 第二轮 Est：论文 AdamW 配置对比

- 单张 GPU，OneDeviceStrategy；保持作者 batch=12、Tiny 双分支、原始街景 512×1024 与卫星 512×512。保持独立 TensorFlow shuffle(seed=626, reshuffle_each_iteration=False) 后 take/skip，2060 对训练、2061 对测试，不使用 Geoloc CSV。
- 使用现有 Keras 2.10 的 `tf.keras.optimizers.experimental.AdamW`，基础学习率 0.03、weight decay=0.05、100 轮。关闭优化器默认 XLA（jit_compile=False），不升级依赖；其权重衰减按 `lr * weight_decay * variable` 更新。
- cosine 调度保留作者公式；预热改为 10 轮，起点沿用公开代码 0.001。每轮 172 批、预热 1720 步、总长度 17200 步。论文及 Est 公开代码未提供 AdamW 排除规则，本轮不额外排除 bias/归一化变量；其余 AdamW 参数采用现有 Keras 默认值。
- 新运行目录保存 `est_train_50.csv`、`est_test_50.csv`、清单哈希及 `est_config.json`。最佳权重仍按最高 val_accuracy 保存；加载后输出完整测试集的 `est_metrics.json`（含 macro/weighted F1、最佳轮次）和关联路径的 `est_predictions.csv`。
- 原 `estimation.txt` 仅保留为逐图展示，它会按原图尺寸过滤样本，不作为完整测试集指标依据。
- 本轮同时更换优化器、学习率、衰减、轮数及 warm-up，并改为单卡；这是论文整套配置与第一轮 SGD 配置的对比，不能将成绩变化全部归因于优化器。

两任务都沿用作者公开代码的训练/测试两部分流程：测试集同时用于选择最佳 checkpoint 和最终报告，没有额外独立验证集。旧 JobID 目录、CSV、模型及结果保留；新版 sbatch 仍用新 JobID 目录隔离运行。

## 十九、提交前将第二轮 Geoloc 改为论文最优比例 6:4（2026-11-04）

用户在提交前确认选择最新版论文表 3 中成绩最好的随机比例。6:4 微调的 R@1/5/10/1%=78.65/97.09/98.85/99.45%，四项均高于 5:5；以表格数字为准。第十八节 Geoloc 的 5:5 设置由本节替代；Est 仍为独立 5:5 单卡 AdamW 实验。

- Geoloc 训练与独立复评统一改为 `val60`，检查 `train_60.csv`、`val_60.csv`。按作者排序配对后 `train_test_split(train_size=0.6, random_state=42)`，4121 对分为 2472 对训练、1649 对测试，不分层。
- 两卡、全局 batch 64、学习率 5e-5、AdamW、10 轮、1 轮预热及其余训练设置不变。调度仍在首次自定义采样之前计算：每轮初始 DataLoader 为 39 批，预热 39 步、总长度 390 步。
- 划分检查器核对 6:4 的样本及行顺序、唯一 ID 和无交叉，保留已有数据，不重新对齐图片或覆盖旧实验。

## 二十、Est 同步改为 6:4 后提交（2026-11-04）

用户随后确认 Est 也使用新版表 7 的 6:4。其 OA=71.38%、macro R=71.73% 是表中最高；macro P=73.13% 低于 5:5 的 73.83%，F1=0.73 与 5:5 相同。因此以 OA/Recall 为选择依据，不称所有指标均最优。

- 保留作者 TensorFlow shuffle(seed=626, reshuffle_each_iteration=False)，训练 2472 对、测试 1649 对；清单文件为 `est_train_60.csv`、`est_test_60.csv`。Geo 与 Est 比例相同，但算法和种子独立，样本清单不共用。
- 单卡、batch 12、AdamW、基础 LR 0.03、weight decay 0.05、100 轮、预热起点 0.001、10 轮预热和 cosine 保持不变；不额外排除 bias 或归一化变量。每轮 206 批、预热 2060 步、总长度 20600 步。
- 最佳权重文件为 `0.6_bestCV.h5`；训练和独立逐图推理均显式传入 `--tr-ratio 0.6`。推理入口不再硬编码 0.5 的划分和权重，避免训练后误读旧比例。
- 本节替代第十八节 Est 的 5:5 参数；第十九节 Geo 的 6:4 设置继续生效。第一轮 SGD 5:5 与本轮 AdamW 6:4 同时改变比例及训练配置，不能解释为只更换优化器的严格消融。

## 二十一、第二轮 6:4 作业已提交（2026-11-04）

本地代码先推送 GitHub，超算 swarm02 工作树检查无修改后通过既有代理转发通路拉取，确认训练提交 `ad24c4969969337f380e0b63f4ce99f199c9a865`。远端 Python/Slurm 语法检查通过，已有 `train_60.csv` 为 2472 行、2472 唯一 ID，`val_60.csv` 为 1649 行、1649 唯一 ID，训练测试无交叉。Geoloc 全部图片及划分顺序核查在计算节点、正式训练前执行；Est 的 6:4 TensorFlow 划分在本次运行启动时重建并保存。没有覆盖旧划分、原图、旧模型或旧输出。

Est 权重缓存预检成功，SHA256=90bf9bfaafd086a831e92edbbfbba9450fd56786c2d39b3ac3f215d0cef54768，默认 Keras 缓存与 scratch 权重指向同一文件。未升级依赖，未运行登录节点训练或 GPU 测试。

| 模块 | JobID | GPU | 全局 batch | 基础 LR | 轮数 | 提交时状态 |
| --- | --- | --- | --- | --- | --- | --- |
| Geoloc | 3971747 | 2 | 64 | 5e-5 | 10 | PENDING (Priority) |
| Est | 3971748 | 1 | 12 | 0.03 | 100 | PENDING (Priority) |

两项独立作业各提交一次，均用 `export CVDISASTER_COMMIT=ad24c4969969337f380e0b63f4ce99f199c9a865` 后 `sbatch --parsable --export=ALL slurm/train_<module>.sbatch`；未取消其他作业。对应远端记录为 `runs/submit-geoloc-3971747.json`、`runs/submit-est-3971748.json`；本机副本在 `.local-cache/job-submissions/`，不上传 GitHub。日志分别为 `/scratch/liuzeyu/heyinxi/logs/geoloc-3971747.log`、`/scratch/liuzeyu/heyinxi/logs/est-3971748.log`。提交成功不等于 GPU 训练验证通过，实际显存、反向传播、单卡 AdamW 和最终指标需待作业运行后核查。

## 二十二、修复 Est 的 CPU 专属缩放算子放置（2026-11-05）

Est `3971748` 在首轮启动时因 GCViT 的 ResizeBicubic 只有 CPU 内核而失败：新增的 OneDeviceStrategy 与 set_soft_device_placement(False) 要求该算子在 GPU 执行。此次仅修复设备执行：训练进程启用软设备放置，保持单卡 GPU 策略，显式核查模型可训练变量仍在 GPU:0；不更改 bicubic 插值算法、网络结构、6:4 划分、batch 12、AdamW、LR 0.03、WD 0.05、100 轮或调度参数，不添加参数衰减排除规则。

Slurm 保留严格 GPU 矩阵预检，再增加 GPU 作用域内 bicubic 缩放的前向和反向检查，以实际覆盖上次失败的算子路径。实际 GPU 检查在新正式作业计算节点执行，本地只做语法及 AST 配置一致性检查。Geo 本次仅分析日志和作者实现，不更改任何代码，不提交新 Geo 作业。

## 二十三、Est 设备修复重提交与 Geo 两轮只读分析（2026-11-05）

超算通过 GitHub 拉取并确认提交 `d14ea6d166fa847e1d3d8e4b89788d9600b0109c`，远端语法检查通过，确认 Geo 路径无改动后仅重新提交 Est 一次：JobID `3979438`，替代失败的 `3971748`，初始状态 PENDING。日志 `/scratch/liuzeyu/heyinxi/logs/est-3979438.log`，提交记录 `runs/submit-est-3979438.json`，本地副本 `.local-cache/job-submissions/est-3979438.json`。不从失败任务续训，使用相同预训练初始化和相同训练参数重新运行；GPU/bicubic 运行检查仍需在正式作业启动后核查。

Geo 两轮对比：第一轮 `3956583` 为 subarea5，训练3854唯一对、测试267唯一对（评价268行），实际600个训练batch、LR=5e-4；第二轮 `3971747` 为随机6:4，训练2472对、测试1649对，实际380个训练batch、LR=5e-5，调度总长度390步。两轮全局batch均64。测试任务及候选库规模不同，不能将81.7164与62.5227直接当作同条件回退。

第二轮zero-shot四项结果与正式论文表3的6:4一致到两位小数，划分逐行检查及最佳权重复评均通过。六个核心模块(model/loss/trainer/dataset/transforms/evaluate)与保存的作者提交6a0b41a经AST核对一致，两轮日志未见NaN或学习率调度警告。

主要待验证假设是优化不足：第二轮训练集最终R@1=60.1537%、测试R@1=62.5227%，loss由3.202降到约1.425；第一轮最终训练R@1=83.2034%、loss=0.926，但数据不同，不能据此做单因素因果判断。论文LR=1e-4与作者公开入口LR=1e-3仍冲突；当前按论文随batch减半为5e-5，相比第一轮低10倍。batch64相对作者128还减少InfoNCE批内负样本；39步与实际38批的调度口径差异约2.6%，单独将16.13个百分点差距归因于它没有证据。未修改或重新提交Geo；后续若验证，应固定6:4划分和batch后单独改变学习率，而非同时改动多个因素。

## 二十四、Geo 仅恢复学习率作对照（2026-11-05）

用户确认实际训练 batch 总数从600降至380主要源于空间划分改为随机6:4后训练唯一对由3854减少到2472；自定义采样每轮保留2432对/64=38批，共10轮。调度仍按首次采样前39批计算，总长度390步，不调整采样或轮数来凑600步。

本次唯一训练参数变更：Geo 基础学习率从5e-5恢复为5e-4。6:4、train_test_split(train_size=0.6, random_state=42)、两卡、全局batch64、10轮、1轮预热、cosine、优化器、损失和其他设置保持不变，Est不改。5e-4对应作者GitHub基础LR=1e-3随batch减半，高于论文正文1e-4的5倍；应标为固定第二轮配置下的学习率对照，不能称为与论文完全相同。

Python AST检查确认训练入口只有lr常量发生变化，Slurm同步更新说明文字。本轮仅改代码并推送GitHub，不提交新Geo作业，既有运行目录与结果保留。

## 二十五、Geo 学习率对照已提交（2026-11-05）

超算拉取并确认训练提交 `8ca60d598d2c306930b1b61d80c098ece846efeb`，语法和参数检查通过；train_60.csv/val_60.csv 的SHA256与上一轮 `3971747` 的运行清单完全一致，2472对训练、1649对测试。仅将Geo学习率改为5e-4，其余6:4、种子42、两卡、全局batch64、10轮、39步预热和390步调度长度保持不变。

新Geo任务JobID `3979539`，提交时PENDING；远端记录 `runs/submit-geoloc-3979539.json`，日志 `/scratch/liuzeyu/heyinxi/logs/geoloc-3979539.log`，本机记录 `.local-cache/job-submissions/geoloc-3979539.json`。各训练产物由新JobID隔离，不覆盖 `3971747`；仍从相同CVUSA预训练权重初始化。Est `3979438` 在g0141运行，未重复提交或修改。

## 二十六、Est 因单卡 OOM 恢复两卡配置（2026-11-05）

Est `3979438` 已失败结束（退出码1:0，09:36:12—10:21:21，45分9秒）：GPU矩阵、AdamW API和bicubic前向/反向预检通过，首轮正式前向传播发生显存不足。TensorFlow GPU配额40174419968字节、实际占用约40159278080字节，再申请约34.45MiB失败；未生成最佳权重或分类报告。单卡承载全局12对，相比此前两卡每卡约6对，显著增加单卡激活显存压力；不能将此次失败解释为AdamW分类效果差。完整日志本机副本在 `.local-cache/run-logs/20261105/est-3979438.log`。

用户确认恢复两卡：训练入口固定两张可见GPU并使用MirroredStrategy，Slurm申请gpu:2、启动参数--gpus 2，逐个检查可训练变量在两张GPU的镜像副本。保留CPU专属bicubic算子的软放置修复，预检两张GPU的实际运算及跨卡归并。

全局batch仍12（每卡约6），6:4、seed626、AdamW、LR0.03、WD0.05、100轮、预热起点0.001、10轮/2060步预热、20600步总调度、模型及输入尺寸均保持不变，不添加衰减排除规则，不因GPU数变化缩放学习率。训练后的逐图展示仍沿用默认主卡路径。AST检查确认上述超参数、划分、模型及调度调用不变；Python、sbatch内嵌Python及shell语法检查通过。

本轮仅更新并推送两卡配置，不自动重新提交Est；Geo代码及正在运行/排队的Geo作业不修改。AdamW两卡的实际训练显存及反向传播仍需新正式作业启动后验证，之前SGD两卡成功不等于本轮AdamW已通过。

## 二十七、Est 两卡 AdamW 作业已提交（2026-11-05）

本地与超算语法/配置检查通过，确认无重复Est活动作业，超算拉取并确认训练提交 `d1ddaddce80f83ec01a71dc437234726dc8af3ef` 后，只提交Est一次：JobID `3988672`，提交时PENDING，替代已失败的 `3979438`。两卡MirroredStrategy、全局batch12/每卡约6、6:4、seed626、AdamW、LR0.03、WD0.05、100轮、10轮warm-up和cosine保持已确认设置，保留CPU专属bicubic算子的软放置修复及GPU权重镜像检查。

提交记录 `runs/submit-est-3988672.json`，本地副本 `.local-cache/job-submissions/est-3988672.json`；日志 `/scratch/liuzeyu/heyinxi/logs/est-3988672.log`；新JobID运行目录隔离旧失败产物。正式作业启动后执行双GPU矩阵、bicubic前向反向及跨卡归并检查，再进入训练；提交成功不等于已完成实际训练显存验证。Geo `3979539` 当时仍在g0179运行，未修改、取消或重复提交。


## 2026-11-05 Est 再提交前的 GPU 容量检查

Est `3988672` 在 g0178 首轮训练 OOM，退出码 1:0；启动日志显示 GPU 0 已被另一 Python 进程占用约 33.5GB，TensorFlow 仅获得约 4.7GiB，GPU 1 获得约 34.1GiB。双卡、bicubic 和通信预检通过，未生成最佳权重或指标；本次失败不足以判断两张空闲 A100 40GB 能否容纳当前训练配置。

22:29 的短时 Slurm 检查作业 `4063272` 在 g0170 看到两张 A100 40GB，逻辑编号 0、1，SLURM_STEP_GPUS=1,2，PCI 地址 81:00.0/C1:00.0；均为 0MiB 占用、40442MiB 空闲、0% 利用率，未列出计算进程。该快照不保留后续训练资源。g0179 当时四张 GPU 均已被 Slurm 分配；g0178 定点检查 `4063123` 未获得资源，未执行 GPU 检查。

用户授权重新提交 Est。启动脚本增加容量检查：在初始化 TensorFlow 前及独立预检结束后，核对两张实际可见显卡的 UUID、显存和计算进程；存在计算进程或任一卡空闲显存不足 38000MiB 时直接失败并打印详情，不终止其他进程、不自动更换设备。两卡、全局 batch12、6:4/seed626、AdamW、LR0.03、WD0.05、100轮、10轮预热及所有训练代码保持不变。Shell/Python 语法及空闲、占用、低显存、其他卡进程、设备数错误等模拟检查通过；正式训练容量仍以新作业实际运行结果为准。

超算经临时 SSH 转发的 GitHub CONNECT 中转完成 fast-forward，同步并确认训练提交 `828ab7490eeb050dd140eeb6164ca82ba95137d6`，Shell 检查通过，确认无活动 Est 后只提交一次：`4065685`。提交时 PENDING（Priority），尚未获得训练节点，未验证正式 batch 的容量。远端记录 `runs/submit-est-4065685.json`，本机记录 `.local-cache/job-submissions/est-4065685.json`，日志 `/scratch/liuzeyu/heyinxi/logs/est-4065685.log`。不固定先前短时检查的节点，必须以本作业实际分配及启动检查为准。


## 2026-11-05 Est 排除占用节点后重新提交

`4065685` 于22:41:32在g0178启动、22:42:02失败，新增容量检查发现GPU0仍有PID387229占用33474MiB、空闲6958MiB，未进入训练。用户授权换GPU尝试；保留全部训练参数及容量检查，经GitHub同步并核对提交`3dc3f00ae1aaeb186bbcfb3cd628e7d06b7d2ddf`，确认无活动Est后使用`sbatch --parsable --export=ALL --exclude=g0178 slurm/train_est.sbatch`只提交一次：`4066740`（22:49:44）。`scontrol show job`确认ExcNodeList=g0178、两卡、全局batch12对应代码未变，提交时PENDING（Priority），尚未分配节点或验证训练。记录为`runs/submit-est-4066740.json`及本机`.local-cache/job-submissions/est-4066740.json`，日志为`/scratch/liuzeyu/heyinxi/logs/est-4066740.log`。节点排除仅针对本次提交，不永久修改Slurm脚本。


## 2026-11-06 Est 恢复作者 GitHub 的 6:4 基线

作业4066740完成100轮后，在最终测试预测中因主机内存超限被Slurm终止（OUT_OF_MEMORY，00:40:22—07:51:27，g0166，MaxRSS=58193.11MiB）；最佳权重来自第1轮，测试accuracy=36.1431%，后续未改善，未生成最终P/R/F1。训练末accuracy=38.11%、测试accuracy=36.14%，与两组中等损伤类占比相符，提示可能退化为多数类预测；尚未用逐样本预测证实，不能将退化直接归因于单独一个超参数。本次两次GPU容量检查均通过，和此前GPU被其他进程占用的故障分开记录。

用户要求以公开GitHub代码为准，只复现6:4。核对来源：https://github.com/tum-bgd/CVDisaster/blob/a303dd8e1d636c36d8688813fb66e9de3842c99c/disaster_perception_mapping/2_cv.py （文件最近变更提交a303dd8e1d636c36d8688813fb66e9de3842c99c）。恢复SGD(lr_schedule)默认参数、LR0.01、200轮、warmup起点0.001、固定10个优化步预热，不再使用AdamW/WD0.05/100轮/10轮预热。保留作者WarmUpCosine公式、batch12、Tiny双分支、SVI512×1024及sat512×512、seed626、shuffle后take/skip、cache/batch/prefetch、损失及val_accuracy选最佳权重；6:4为2472训练/1649测试，每轮206步，总调度41200步。

工程差异继续明确保留：两卡MirroredStrategy及作用域、CPU专属bicubic软放置、超算数据/权重路径、GPU容量检查、24小时作业上限、划分和完整指标文件。作者3_estimation.py硬编码0.5，继续改为读取--tr-ratio=0.6及对应最佳权重。为处理本次报告阶段OOM，训练结束后释放训练数据缓存，预测改为predict_on_batch逐批保留标签和类别索引；不参与训练、最佳权重选择或改变测试样本。未改Geo，未提交新训练。

验证：与公开代码逐项AST比对WarmUpCosine、GetLabel、LoadImg、ProcessPath、ConfigureData、模型创建、调度、SGD、损失、compile、fit、checkpoint配置均匹配；Shell及内嵌Python语法通过。两卡执行及训练后内存改善仍需超算真实运行验证，本地无TensorFlow运行测试。


## 2026-11-06 Est GitHub SGD 6:4 基线重新提交

逐项比对成功作业3956738的训练提交5210411：WarmUpCosine、数据处理/缓存、SGD及默认参数、损失、模型与分类头、compile、fit、checkpoint选择、batch12、输入尺寸、seed626、200轮及10步预热均通过AST一致性检查。6:4相较成功的5:5增加训练样本及每轮步数，总调度按206×200=41200计算；保留两卡作用域、软放置及记录适配。独立预检新增两卡微型模型SGD更新和predict_on_batch处理5样本尾批；实际执行仍待分配计算节点。

提交前确认个人无其他活动作业；短时GPU检查4087094因无即时资源未获得显卡，未读到实时占用。正式任务继续排除g0178，保留初始化前及预检后的两次进程/显存检查，任一卡已有计算进程或空闲少于38000MiB直接失败。经GitHub拉取并确认fd3c44f8b1faa7d48ceff37a92401c44da683367，远端Shell/Python/SGD配置检查及权重缓存SHA256验证通过；未改环境依赖或Geo。

只提交一次正式Est：4087098，2026-11-06 12:41:59，初始PENDING（Priority）；scontrol确认gpu2/cpu8/mem60000MiB/24小时/ExcNodeList=g0178，无作业依赖。运行记录runs/submit-est-4087098.json、本机.local-cache/job-submissions/est-4087098.json，日志/scratch/liuzeyu/heyinxi/logs/est-4087098.log。当前无GPU分配，不能称实际训练或主机内存改善已验证。


## 2026-11-06 Est best/latest 与显式断点续训

按用户要求取消4087098，Slurm确认CANCELLED，未进入训练。新增est_checkpoints.py：best按测试accuracy严格提高时原子替换；每轮将全部模型权重、SGD优化器状态/iterations、已完成轮次、历史最佳分数/轮次保存到latest，完整写入后原子发布current.json并清理旧快照。latest内部通过硬链接保留对应best，测试了下一轮best写到一半中断时仍能恢复上次完整状态；scratch原子重命名/硬链接检查通过。

新实验默认从ImageNet预训练初始化，不读取其他任务。仅显式传入`sbatch ... slurm/train_est.sbatch --resume-from 旧训练输出目录`才续训，忽略历史续训环境变量。新JobID保存独立输出，不覆盖旧目录；恢复时严格核对训练配置、划分哈希、权重结构和TensorFlow版本。总目标仍200轮，已完成训练时直接测试best；未完整保存轮次重跑，未保存数据迭代器/完整Dropout随机流，不承诺逐位一致。旧版best-only任务不可假装具备latest。使用说明见EST_CHECKPOINTS.md。

SGD、LR0.01、batch12、200轮、10步预热、两卡、6:4及总调度41200步不变，已与成功实验5210411逐项AST核对。Shell及Python语法通过。真实TensorFlow检查在hpxg的两逻辑CPU镜像策略下运行：4087174因Lustre读取等待在框架初始化阶段触及20分钟时限；4087223复测输出EST_CHECKPOINT_ROUNDTRIP_OK，验证模型/optimizer恢复、下一步SGD更新与未中断路径一致、历史best保留、未发布快照忽略及配置不匹配拒绝。检查期间仅在计算节点预读框架文件以缓解共享文件系统I/O，不修改环境依赖；GPU镜像路径仍由正式作业独立微模型预检验证。
