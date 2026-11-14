# Geo CVFT seed=42 正式训练提交记录

日期：2026-11-12。方案编号：`geo_cvft_refine`。

用户授权检查超算环境后提交一个正式训练作业。本次只训练改进模型，已有基线权重用于评价和结果比较。

- JobID：`4277758`，作业名称：`geo-cvft-s42`。
- 提交时间：2026-11-12 13:42:55（Asia/Shanghai）。
- 使用的代码提交：`c128a4fea610de717da2fcc954a44fe64aeba868`，通过 Git bundle 同步到超算，工作目录通过 fast-forward 更新。
- 提交回执状态：`PENDING`，原因为 `Priority`，`Restarts=0`，无依赖作业。
- 提交后账号队列中只有本作业。
- 启动时间：2026-11-12 13:43:24（Asia/Shanghai）；核对时状态为 `RUNNING`，节点 `g0176`，`Restarts=0`。
- 启动日志确认 `MODE=train` 和上述代码版本，已识别两张 NVIDIA A100-SXM4-40GB。核对时处于初始化阶段，日志尚无 Python 报错。
- 回执文件：`/scratch/liuzeyu/heyinxi/CVDisaster/.local-cache/geo-reuse-checks/submission-c128a4f.jobid`。
- 日志：`/scratch/liuzeyu/heyinxi/logs/geo-cvft-4277758.log`。
- 运行目录：`/scratch/liuzeyu/heyinxi/runs/geo_cvft_refine-seed42-4277758`。
- 结果目录：`/scratch/liuzeyu/heyinxi/runs/geo_cvft_refine-seed42-4277758/output`。

训练参数：seed=42，新增模块初始化 seed=42，10 个 epoch，训练 batch=64，评价 batch=128，AdamW 学习率 `5e-4`，label smoothing=0.1。InfoNCE、AMP、梯度裁剪、训练集与测试集 Recall、自定义采样和调度计算沿用原 geo 复现，best 初值为 0。Sinkhorn 固定 30 次、温度 1.0，融合初始比例各为 0.5，并通过训练学习。

资源：`a100x4` 分区、`wanyi` 账号、一个节点、两张 A100、8 个 CPU、60,000 MiB 主机内存，时间上限 24 小时。提交入口为 `ME CVDisaster/scripts/experiments/submit_geo_cvft.sh`，使用纯 `--export=ALL`，并检查同名活动作业。

同一作业内执行训练前评价、10 个 epoch 的训练与每轮评价、best 保存加载检查，以及已有基线的评价和比较。本次不另行提交独立验收或基线训练作业。

超算已有环境中的 10 项 CPU 函数检查已通过，覆盖原 InfoNCE 损失与梯度一致性、原训练函数更新一致性、可变 batch、配置一致性、新模块初始化及保存加载。

提交前完整训练入口的依赖导入通过，确认直接调用原 `train` 和 `InfoNCE`。实际环境版本为 PyTorch `1.13.1+cu116`、torchvision `0.14.1+cu116`、timm `0.8.19dev0`、transformers `4.27.0`、albumentations `1.3.0`、OpenCV `4.11.0`、pandas `2.3.3`。两份权重文件和训练、测试清单均可读取，清单分别包含 2472 对和 1649 对图片。

存储检查：共享文件系统可用约 105 TB；用户已用 996.1 GB，配额及上限均为 3 TB；文件数量 911615，配额及上限均为 1000000。`a100x4` 分区可用，提交时无空闲节点。环境导入检查曾等待共享存储读取，随后正常完成。
