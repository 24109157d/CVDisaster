# geo实验1

新实验为 `geo_partial_match_gate`，方案见 [困难候选局部判断与动态融合实验方案](docs/Geo_部分匹配与动态融合实验方案.md)。

全局部分直接使用复现的 `TimmModel`、训练函数、InfoNCE 与全局难负采样。新增模块读取原有评价前向的两层 FP32 缓存，训练候选对部分匹配、局部评分和查询级融合。局部梯度不进入原有参数，全局与新增参数分别使用 optimizer、scheduler、GradScaler 和随机状态。

新实验入口：`train_partial.py`。训练配置：`configs/geo_partial_match_gate.json`。入口接受 `--mode train` 或 `--mode evaluate`、`--output`；评价同时提供 `--checkpoint`，支持 `best_global.pth`、`best_fusion.pth` 和 `last.pth`。

对应作业定义为 `scripts/experiments/geo_partial_match_gate.sbatch`。正式训练和评价要求 Slurm 的两张 A100 环境，使用已同步提交的独立源码副本。实现与本地检查不会自动提交任务。

在本目录执行 `python tests/test_partial_matching.py` 检查真实模块的数值、梯度、生产形状、原 ConvNeXt 前向一致性、随机状态隔离、整轮局部训练、候选采样、checkpoint 分组和保存加载。测试使用明确的数值输入；真实 CVIAN、GPU、AMP 与资源验收结果分别记录。

已有 `geo_cvft_refine` 的运行说明见 [CVFT 实验记录](docs/Geo_CVFT_空间特征修正实验方案.md)，入口为 `train.py`、`evaluate.py`、`validate.py`、`compare.py`，配置为 `configs/geo_cvft_refine.json`，旧模型和结果继续按对应配置读取。

原有回归检查使用 `python tests/test_training_design.py`。
