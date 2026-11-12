# Geo CVFT seed=42 正式训练提交记录

日期：2026-11-12。方案编号：`geo_cvft_refine`。

用户授权检查超算环境后提交一个正式训练作业。本次只训练改进模型，已有基线权重用于评价和结果比较。JobID、最终代码版本和提交状态在取得回执后填写。

训练参数：seed=42，新增模块初始化 seed=42，10 个 epoch，训练 batch=64，评价 batch=128，AdamW 学习率 `5e-4`，label smoothing=0.1。InfoNCE、AMP、梯度裁剪、训练集与测试集 Recall、自定义采样和调度计算沿用原 geo 复现，best 初值为 0。Sinkhorn 固定 30 次、温度 1.0，融合初始比例各为 0.5，并通过训练学习。

资源：`a100x4` 分区、`wanyi` 账号、一个节点、两张 A100、8 个 CPU、60,000 MiB 主机内存，时间上限 24 小时。提交入口为 `ME CVDisaster/scripts/experiments/submit_geo_cvft.sh`，使用纯 `--export=ALL`，并检查同名活动作业。

同一作业内执行训练前评价、10 个 epoch 的训练与每轮评价、best 保存加载检查，以及已有基线的评价和比较。本次不另行提交独立验收或基线训练作业。

超算已有环境中的 10 项 CPU 函数检查已通过，覆盖原 InfoNCE 损失与梯度一致性、原训练函数更新一致性、可变 batch、配置一致性、新模块初始化及保存加载。
