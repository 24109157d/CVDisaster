# Geo CVFT seed=42 正式训练提交记录

日期：2026-11-11。方案编号：`geo_cvft_refine`。

## 作业与源码

- 用户已确认本次任务与参数，并明确要求提交。
- JobID：`4271513`。
- 作业名称：`geo-cvft-s42`。
- 训练代码提交：`1b25ba83a1dba4c7fd60cc68480d515d9252e307`。
- 本地代码已推送 GitHub，超算通过 `git pull --ff-only origin main` 同步后提交。
- 作业脚本：`ME CVDisaster/scripts/experiments/geo_cvft_refine.sbatch`。
- 提交入口：`ME CVDisaster/scripts/experiments/submit_geo_cvft.sh`。
- 提交回执：`/scratch/liuzeyu/heyinxi/runs/submit-geo-cvft-seed42-1b25ba8.txt`。
- 日志：`/scratch/liuzeyu/heyinxi/logs/geo-cvft-4271513.log`。
- 运行目录：`/scratch/liuzeyu/heyinxi/runs/geo_cvft_refine-seed42-4271513`。
- 输出目录：`/scratch/liuzeyu/heyinxi/runs/geo_cvft_refine-seed42-4271513/output`。

## 参数

| 项目 | 设置 |
| --- | --- |
| 正式训练模型 | `geo_cvft_refine` |
| seed | 42 |
| 训练轮数 | 10 个 epoch，每轮 38 次更新，共 380 次更新 |
| 分区与账号 | `a100x4`、`wanyi` |
| 节点与 GPU | 一个节点、两张 A100 |
| CPU 与主机内存 | 8 个 CPU、60,000 MiB |
| 时间上限 | 24 小时 |
| 全局训练 batch | 64 对图片 |
| 全局评估 batch | 128 幅图片 |
| DataLoader workers | 4 |
| optimizer | AdamW，LR `5e-4`，betas `(0.9,0.999)`，eps `1e-8`，weight decay `0.01` |
| scheduler | cosine，预热 39 步，总长度 390 步 |
| 训练目标 | 最终融合向量的双向 InfoNCE，label smoothing `0.1` |
| AMP 与梯度 | AMP，GradScaler 初值 1024，逐元素梯度裁剪 100 |
| CVFT | 温度 1.0，Sinkhorn 20 次，行列误差阈值 `1e-3` |
| 融合 | `fusion_logit` 初值 0，两个分支初始比例各 0.5，比例参与训练 |

## 数据与环境检查

数据目录为 `/scratch/liuzeyu/heyinxi/dataset/m1749324/CVIAN`，使用已有 `train_60.csv` 和 `val_60.csv`。初始权重为 `/scratch/liuzeyu/heyinxi/pretrained/weights_e40_98.6830.pth`。

已确认作业 `3979539` 的已有 best checkpoint：`/scratch/liuzeyu/heyinxi/runs/geoloc-3979539/Author CVDisaster/geolocalization/cvdisaster/convnext_base.fb_in22k_ft_in1k_384/111318/weights_e10_79.8059.pth`。

环境为 `/scratch/liuzeyu/anaconda3/envs/cvdisaster_geoloc`，Python 3.10.22、PyTorch `1.13.1+cu116`、torchvision `0.14.1+cu116`、timm `0.8.19.dev0`，版本已通过 metadata 核对。提交前账户队列为空。2026-11-11 21:45 的存储检查为：`/scratch` 已使用约 996.1 GiB，配额 3 TiB，文件数量 911,432，上限 1,000,000。

本地 Python 编译、Pyflakes 和作业脚本语法检查通过。完整运行结果由本次正式作业产生。

## 同一作业内的执行内容

1. 评价初始改进模型，生成训练集难负候选。
2. 完成改进模型 10 个 epoch 的训练，每轮结束后评价并按测试 R@1 保存 best。
3. 保存 last，重新加载 best，计算完整测试集指标与逐查询预测。
4. 读取作业 `3979539` 的已有 best checkpoint，执行当前和作者的评价流程，基线参数更新次数为零。
5. 保存基线逐查询预测，生成四项 Recall 的百分点差与正确性比较。

提交后返回状态为 `PENDING (Priority)`，时间上限为 `1-00:00:00`。本次已提交的计算作业数量为一个。提交进程提前导出训练参数，清除用户环境获取选项，并使用纯 `--export=ALL` 将当前环境传递给 Slurm。

2026-11-11 21:45 核对：作业已于 21:42:48 在 `g0153` 开始运行，状态为 `RUNNING`，`Restarts=0`，分配两张 A100、8 个 CPU 和 60,000 MiB 主机内存。

## 提交命令

```bash
export CVDISASTER_COMMIT=1b25ba83a1dba4c7fd60cc68480d515d9252e307
bash "ME CVDisaster/scripts/experiments/submit_geo_cvft.sh"
```
