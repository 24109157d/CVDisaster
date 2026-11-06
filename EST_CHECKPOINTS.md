# Est 检查点与续训

新实验默认从 ImageNet 预训练权重开始，不自动读取其他任务的 checkpoint。只有同一实验因超算异常中断、且明确指定 `--resume-from` 时，才恢复该任务的 latest。

## 保存内容

| 检查点 | 保存时机 | 用途 |
|---|---|---|
| `0.6_bestCV.h5` | 测试集 accuracy 严格提高时替换 | 最终测试和逐图展示 |
| `latest/current.json` 指向的完整快照 | 每个完整训练轮次结束时替换 | 恢复模型、优化器、迭代步、已完成轮次、历史最佳分数及轮次 |

latest 先写完整目录，再原子发布指针，发布成功后清理旧快照。快照内保留当时 best 的硬链接，避免中断在保存过程中导致 best/latest 不一致。正常情况下只保留一个已发布的 latest；中断时遗留的临时文件不作为恢复来源。

## 提交新实验

在超算仓库同步代码、确认资源后，按正常方式提交，不传续训参数：

```bash
export CVDISASTER_COMMIT=$(git rev-parse HEAD)
unset SBATCH_GET_USER_ENV SLURM_GET_USER_ENV SBATCH_EXPORT
sbatch --export=ALL slurm/train_est.sbatch
```

日志应显示 `EST_START_MODE=fresh_pretrained`。脚本不读取历史的 `CVDISASTER_EST_RESUME_FROM` 环境变量。

## 续跑同一个中断实验

先确认旧任务已停止、旧任务确实保存了本版本的 latest，且沿用相同代码、环境、数据划分及训练参数，再显式传旧训练输出目录：

```bash
export CVDISASTER_COMMIT=$(git rev-parse HEAD)
unset SBATCH_GET_USER_ENV SLURM_GET_USER_ENV SBATCH_EXPORT
sbatch --export=ALL slurm/train_est.sbatch --resume-from "/scratch/liuzeyu/heyinxi/runs/est-旧任务编号/Author CVDisaster/disaster_perception_mapping"
```

新 JobID 保存独立输出，旧目录不会被修改。恢复后总目标仍是 200 轮，而非再训练 200 轮；若 200 轮已完成、仅测试中断，则直接加载 best 测试。配置、划分、权重结构或 TensorFlow 版本不一致时直接报错，不回退到 best 或其他实验。

恢复粒度为完整轮次，最近未完整保存的轮次需要重跑。模型和优化器状态会恢复，数据迭代器及完整 Dropout 随机流不在检查点内，不保证与未中断运行逐位相同。旧版仅有 best 的任务不能恢复其未保存的 latest。本功能不会自动重提作业。
