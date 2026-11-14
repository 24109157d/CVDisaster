# Geo CVFT seed=42 正式训练提交记录

日期：2026-11-13（Asia/Shanghai）。

用户授权按当前方案提交一次正式训练。JobID 为 `4290487`，作业名称为 `geo-cvft-s42`，提交时间为 2026-11-13 08:41:26。提交回执状态为 `PENDING`，无依赖作业，`Restarts=0`；提交后账号队列中只有本作业。

启动时间为 2026-11-13 08:41:27。核对状态为 `RUNNING`，节点为 `g0167`，`Restarts=0`。启动日志确认 `MODE=train`、上述运行目录和代码提交，并识别两张 NVIDIA A100-SXM4-40GB。核对时处于启动检查阶段，日志尚无 Python 报错。

## 代码与输出

- 训练代码提交：`e4804f35bae339d358774da80431677c356aad6c`。
- 源码分支：`codex/geo-cvft-20261113`，使用独立 Git index 生成训练快照，当前本地工作分支保持原值。
- 超算通过 Git bundle 获取此提交，运行时使用 `git archive` 生成独立源码目录，超算仓库的工作目录保持原值。
- Git bundle SHA256：`abf8d1bcc46bf05d523ba15db79d8b3ccdb3192c8c43c8c9e2c3f2defc2ee436`。
- 提交回执：`/scratch/liuzeyu/heyinxi/CVDisaster/.local-cache/geo-submit-20261113/submission-e4804f35bae339d358774da80431677c356aad6c.jobid`。
- 日志：`/scratch/liuzeyu/heyinxi/logs/geo-cvft-4290487.log`。
- 运行目录：`/scratch/liuzeyu/heyinxi/runs/geo_cvft_refine-seed42-4290487`。
- 输出目录：`/scratch/liuzeyu/heyinxi/runs/geo_cvft_refine-seed42-4290487/output`。

## 训练设置

- 同一 CVUSA 初始权重，geo 原有参数沿用复现代码与 optimizer 设置，基础学习率 `5e-4`、weight decay `0.01`。
- 新增空间参数组基础学习率 `5e-4`，`fusion_logit` 参数组基础学习率 `1e-2`、weight decay 为 0。
- 全局、空间、融合三项 InfoNCE 之和，三项系数均为 1，label smoothing 为 0.1。
- 初始融合比例为全局 70%、空间 30%，后续通过训练学习。
- Sinkhorn 至少执行 30 次，每增加 10 次检查完整 batch；全部样本误差不超过 `1e-3` 时结束，最多 500 次，达到上限后使用当前矩阵继续计算。
- seed=42，新增模块初始化 seed=42，训练 10 个 epoch，训练 batch=64，评价 batch=128，沿用原有 warmup、cosine、AMP、梯度裁剪、难负采样和 best 选择规则。
- 使用已有 geo 作业 `3979539` 的 checkpoint 执行最终比较，该权重的参数更新次数为零。

## 资源与检查

资源为 `a100x4` 分区、`wanyi` 账号、一个节点、两张 A100、8 个 CPU、60,000 MiB 主机内存，时间上限 24 小时。

提交前确认两份权重可读取，训练与测试 CSV 分别包含 2472 对和 1649 对图片，首尾图片可读取。共享文件系统可用约 101 TB；账号使用 996.8 GB，配额及上限为 3 TB；文件数量为 912025，配额及上限为 1000000。`a100x4` 分区可用。

本地 14 项 CPU 数值检查再次全部通过。登录节点依赖导入等待 Lustre 读取，已结束该检查进程。正式作业在同一计算节点内执行 `CUDA_VISIBLE_DEVICES="" python -u tests/test_training_design.py`，检查成功后执行正式训练；未另外提交检查作业。

提交入口使用 `--export=ALL`，检查同名活动作业；外层提交脚本使用文件锁及持久回执，确保本次只提交一个作业。提交成功表示 Slurm 已接收任务，CPU 检查与训练进展以本作业日志为准。
