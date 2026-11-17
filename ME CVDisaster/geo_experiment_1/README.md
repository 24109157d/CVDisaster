# geo实验1

当前实验为 `geo_partial_match_fixed`，采用固定局部 40%、全局 60% 融合。主要说明见 [实验方案](docs/实验方案.md)，其他记录按用途归类，见 [文档目录](docs/README.md)。

第一次成功改进实验为作业 `4316915`，最终融合 R@1 为 **84.2935%**，比原 geo 的 **79.8059%** 提高 **4.4876 个百分点**。汇报使用 [首次成功改进汇报](docs/首次成功改进汇报.md)，原始数值和后续分析使用 [结果文件](results/4316915/README.md)。

| 查看内容 | 入口 |
|---|---|
| 汇报对比清单与流程图 | `docs/首次成功改进汇报.md` |
| 当前完整设计与参数依据 | `docs/实验方案.md` |
| 实际训练参数 | `configs/geo_partial_match_fixed40.json` |
| 当前训练与评价 | `train_partial.py` |
| 当前 Slurm 脚本 | `scripts/experiments/geo_partial_match_fixed40.sbatch` |
| 局部结构、候选选择、训练流程 | `geolocalization/local_matching.py`、`local_retrieval.py`、`partial_experiment.py` |
| 原始实验结果与逐查询分数 | `results/4316915/` |
| 每次实际运行的源码和成绩 | `docs/records/` |

全局部分直接使用复现的 `TimmModel`、训练函数、InfoNCE 与全局难负采样。新增模块读取原有评价前向的两层 FP32 缓存，通过 `0.5L_local+0.5L_fusion` 学习候选对部分匹配和局部评分。训练、评价与推理均使用 `0.6s_global+0.4s_local`。局部梯度不进入原有参数，全局与新增参数分别使用 optimizer、scheduler、GradScaler 和随机状态。

局部损失温度固定为 `0.07`，融合损失温度固定为 `0.05`，与原全局温度分别管理。局部困难候选在每个 batch 按当前评分重新选择，Top-20 分数同时用于局部与融合损失；原 geo 的训练数据划分、图像 batch、采样与参数更新保持复现设置。

当前入口为 `train_partial.py`，训练配置为 `configs/geo_partial_match_fixed40.json`。入口接受 `--mode train` 或 `--mode evaluate`、`--output`；评价同时提供 `--checkpoint`，支持本实验的 `best_global.pth`、`best_fusion.pth` 和 `last.pth`。

对应作业定义为 `scripts/experiments/geo_partial_match_fixed40.sbatch`。正式训练和评价要求 Slurm 的两张 A100 环境，使用已同步提交的独立源码副本。实现与本地检查不会自动提交任务。

在本目录执行 `python tests/test_partial_matching.py` 检查真实模块的数值、梯度、生产形状、原 ConvNeXt 前向一致性、随机状态隔离、整轮局部训练、候选采样、checkpoint 分组和保存加载。测试使用明确的数值输入；真实 CVIAN、GPU、AMP 与资源验收结果分别记录。

已有 `geo_cvft_refine` 的运行说明见 [CVFT 历史方案](docs/archive/Geo_CVFT_空间特征修正实验方案.md)，入口为 `train.py`、`evaluate.py`、`validate.py`、`compare.py`，配置为 `configs/geo_cvft_refine.json`，旧模型和结果继续按对应配置读取。作业 `4309752` 的动态融合 checkpoint 使用该作业目录保存的源码副本读取。

这些历史入口及其模块保留旧模型读取和回归检查用途；当前 `runtime.py` 仍引用其中的公共模型工具。后续固定比例实验从 `train_partial.py` 开始，生成结果保存在独立运行目录。

原有回归检查使用 `python tests/test_training_design.py`。
