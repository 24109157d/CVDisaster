# Geo 困难候选局部判断实验提交记录

日期：2026-11-14，时区 Asia/Shanghai。

## 作业与源码

- 有效作业：`4309752`，名称 `geo-partial-match`。
- 已核对状态：`COMPLETED`，节点 `g0153`，退出码 `0:0`，耗时 `04:12:47`。
- Slurm 启动时间：2026-11-14 12:31:34。
- 代码提交：`edce05ead3d6e6bae65b7f77685dc609334331b3`。
- 独立分支：`codex/geo-partial-20261114`。
- 来源提交：`0f439e0d38e6cb1724a2ae7f41826365386d1033`。
- 本地与服务器原有工作目录保持原样，运行使用该提交的独立源码副本。
- 本次源码提交未修改 `Author CVDisaster/geolocalization`，全局原模型、InfoNCE、训练函数及计算流程直接调用已有复现实现。

作业目录为 `/scratch/liuzeyu/heyinxi/runs/geo_partial_match_gate-train-4309752/`，源码保存在其中的 `source/`，输出在 `output/`。日志为 `/scratch/liuzeyu/heyinxi/logs/geo-partial-match-4309752.log`。

## 提交设置

| 参数 | 设置 |
|---|---|
| 分区、账号 | `a100x4`、`wanyi` |
| GPU、CPU、内存 | 两张 A100、8 个 CPU、60000 MB |
| 时间上限 | 24 小时 |
| 训练轮数、seed | 10、42 |
| 原全局 batch、图像评价 batch | 64、128 |
| 原 geo 学习率 | `5e-4`，原 AdamW、cosine、39 步 warmup、390 步调度 |
| 原有参数梯度 | 仅 `L_global` |
| 新增模块、门控 MLP 学习率 | `5e-4`、`5e-4` |
| `fusion_logit` 学习率 | `1e-2` |
| 新增模块损失 | `0.5L_local+0.5L_fusion` |
| 初始局部预测比例 | 5%，训练中按查询学习 |
| 局部候选 | 正确 1、全局困难 4、局部困难 4、随机 3 |
| 重新排序范围 | 全局前 20 个候选 |
| 特征与匹配 | 双层特征，128 维，4 层 self/cross attention，带 dustbin |
| Sinkhorn | 最少 30、每 10 次检查、最多 500、阈值 `1e-3` |

完整参数以源码中的 `configs/geo_partial_match_gate.json` 为准。全局与新增模块分别管理 optimizer、scheduler、GradScaler 和随机状态；输出 `best_global.pth`、`best_fusion.pth` 和 `last.pth`。

## 提交前核验

服务器 Python 为 3.10.22、PyTorch 为 1.13.1+cu116、torchvision 为 0.14.1+cu116、timm 为 0.8.19dev0、transformers 为 4.27.0，原预训练缓存可读取。A100 分区正常，存储空间和用户配额满足本次输出需要。

CVUSA 文件 SHA256：`739cf00d303836fba1195f1bb8d54c4b13c53926ce3212fc9ceced335fc59a23`。

训练清单 SHA256：`53286cb98749755f7e04d1b8b2b4a64a00331389ea75dff6d1a99b5d99378ccd`。

测试清单 SHA256：`f37b6f4e18f001d1b2e46be92fd819275ac6c45581ec3eca30adf7c08d88548d`。

本地新增模块 12 项检查通过，原有 14 项回归检查通过，作业脚本语法检查通过。本次提交前再次运行新增 12 项检查，全部通过。源码同步后核验提交标识、入口存在性及服务器脚本语法。

有效提交使用提前导出的 `CVDISASTER_COMMIT`、`CVDISASTER_MODE=train`，并以 `--export=ALL` 传递环境。Slurm 未能读取用户环境的作业 `4309749` 已取消，耗时为零，未进入训练脚本。

## 完成结果

主融合 checkpoint 为第 10 轮，保存加载后排名核验通过。

| 评价方式 | R@1 |
|---|---:|
| 原 geo 与本次全局分支 | 79.8059% |
| 训练所得动态融合 | 80.1698% |
| 同一 checkpoint，推理局部比例固定 30% | 83.9903% |
| 同一 checkpoint，推理局部比例固定 40% | 84.0509% |

动态融合纠正 7 个全局错误、新增 1 个错误。固定 40% 相比原 geo 净增加 70 个正确查询；这两个数量对应不同的融合方式。固定比例成绩复用本作业训练后的参数，未执行固定比例训练。最终查询平均局部权重为 2.5989%，范围约为 2.51% 至 2.64%。

结果文件为作业目录下 `output/result.json` 和 `output/best_queries.json`。checkpoint SHA256 为 `96ec5397d2188143c319b4bc859f7c8c685b1b6676d77bf5f3cbe50d71d7abea`。最大 RSS 为 `24014952K`。
