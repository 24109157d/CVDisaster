# 改进实验一结果文件

作业 `4316915`，源码提交 `591f31003f01977beb6c21e6ee6163d8b56ea81f`。JSON 和 JSONL 文件从该作业输出原样保存，评分数组从同一作业的 `best_queries.json` 提取，便于后续比较和检查。

| 文件 | 用途 |
|---|---|
| `result.json` | 最佳完整模型重新加载后的最终成绩、全局核验指标、97 个纠错与 23 个新增错误 |
| `history.json` | 10 轮全局与融合指标、更新计数、样本顺序与全局参数保护检查 |
| `run.json` | 完整配置、环境、数据清单 SHA256、原全局源码 SHA256 和输入变换 |
| `local_steps.jsonl` | 新增模块每步损失、温度、学习率、迭代误差和实际更新状态 |
| `retrieval_scores.npz` | 最佳模型的逐查询 ID、候选 ID、全局与局部分数、全局与融合排名、固定权重；用于分析具体错误和比较其他推理比例 |

使用 `numpy.load(path, allow_pickle=False)` 读取评分文件，其字段为 `ids`、`candidate_ids`、`global_scores`、`local_scores`、`global_ranks`、`ranks`、`beta`。图片 ID 和排名使用 int64，分数使用原 FP32 精度。排名字段从零开始计数，`rank=0` 表示第一名正确。

`candidate_ids`、`global_scores` 和 `local_scores` 的形状为 `[1649,20]`，对应每个查询的全局前 20 个候选；其余字段为长度 1649 的数组。

完整 `best_queries.json` 包含额外匹配统计量，原件与模型权重保留在 `/scratch/liuzeyu/heyinxi/runs/geo_partial_match_fixed40-train-4316915/output/`。该目录同时保留每轮候选清单和每轮逐查询预测。

完整模型为 `best_fusion.pth`，SHA256：`23cf8789b0b04b5e564741972dcaf9c38eef46d16a3037db77c321a513d0bce1`。全局独立模型为 `best_global.pth`，最终轮次完整模型为 `last.pth`。
