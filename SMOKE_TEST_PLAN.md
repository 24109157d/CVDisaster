# CVDisaster GPU Smoke Test 方案（已确认，2026-10-28）

方案编写阶段仅做只读调查、本地脚本生成和静态检查。用户已于 2026-10-28 确认执行：先挂起两条旧完整训练任务，再提交 Geoloc；检查通过后才提交 Est，全程不重新安装依赖。实际 JobID 和结果另行记录。

## 1. 已核对的前置条件

- 现有环境：`/scratch/liuzeyu/anaconda3/envs/cvdisaster_geoloc`（PyTorch 1.13.1+cu116）；`/scratch/liuzeyu/anaconda3/envs/cvdisaster_est`（TensorFlow 2.10.1）。
- 原始数据：`/scratch/liuzeyu/heyinxi/dataset/m1749324/CVIAN`，两视角均有 `0_MinorDamage`、`1_ModerateDamage`、`2_SevereDamage` 三个类别目录。
- Geoloc 的正式 `train_subarea5.csv`、`val_subarea5.csv` 尚不存在。Smoke 不调用会向原始数据目录写入的 `0_prepare_splits_and_data.py`。
- 现有权重与缓存：CVUSA `pretrained/weights_e40_98.6830.pth`；ConvNeXt 的固定 Hugging Face 缓存；GCViT `pretrained/keras/datasets/gcvit_tiny_weights.h5`。均使用已有文件，校验失败或缺失直接退出，不下载替代文件。
- 旧完整训练任务 `3780573`、`3780728` 在本次只读查询时仍为 `PENDING (AssocGrpCpuLimit)`。它们保留着完整训练指令，可能在配额释放后启动。建议确认 smoke 提交方案时，同时确认挂起或取消这两个旧任务；本轮不执行这些操作。

## 2. Geoloc 测试内容

入口：`scripts/smoke/geoloc.py`。复用作者的 `TimmModel`、数据集类、训练/评估图像变换、`InfoNCE` 和 `trainer.predict`，不执行完整训练入口。

1. 强制 PyTorch 看到一张 GPU，在 GPU 上执行 16×16 矩阵乘法，核对结果。
2. 按类别轮流选 4 对真实图像，按类别目录和同名文件配对，检查数字 ID、图像尺寸和 GeoJSON 方位角。
3. 仅在本次输出的 `sample-data/` 中复制卫星图像、生成 4 张按方位角对齐的街景，以及两行训练 CSV、两行评估 CSV。训练与评估 ID 不重叠，不代表正式空间划分。
4. 初始化 `convnext_base.fb_in22k_ft_in1k_384`，使用已有 ImageNet 初始化缓存并加载已有 CVUSA 权重。严格核对参数键；不兼容时报告错误，不静默忽略参数。
5. 训练恰好一个 batch：2 对图像，卫星 384×384、街景 140×768，保持作者裁剪与增强方式；位置标签为 `torch.int64`。使用作者对称 InfoNCE、交叉熵 label smoothing 0.1、AMP、AdamW（lr=0.001）和梯度裁剪。
6. 检查 loss 有限且非零、所有可训练参数梯度存在且有限、至少一个梯度非零、一次 optimizer step 后骨干参数实际改变。
7. 对另外 2 对样本提取归一化特征，计算 2×2 相似度矩阵和 Recall@1。使用作者特征提取函数；微型候选集不调用原评估器中的 `R//100` top1% 项，以免两个候选产生 0 截止位。该 Recall@1 只验证流程，不能作为定位实验结论。

batch 不设为 1，因为单对样本的对比学习没有负样本，无法有效验证非退化 loss。

## 3. Est 测试内容

入口：`scripts/smoke/est.py`。复现作者 `2_cv.py` 中双分支 GCViT Tiny、分类头、图像处理和稀疏交叉熵，不导入其会启动完整训练的顶层代码。

1. 强制 TensorFlow 看到一张 GPU，关闭静默设备回退，执行 GPU 矩阵乘法。
2. 各类别选 1 对，共 3 对真实图像；核对街景 512×1024、卫星 512×512，标签为 int64 的 0/1/2。图像按照作者 resize，保留 float32 的 0..255 数值范围，不另加归一化。
3. 两个 GCViT Tiny 分支均保留原始分辨率，`pretrain=True, resize_query=True`，加载已有缓存；使用作者 Dense 64 / LeakyReLU / Dropout 0.2 / Dense 3 softmax 分类头，保持 float32。
4. 一对图像组成唯一训练 batch，通过 eager `GradientTape` 执行前向、稀疏交叉熵、反向传播、一次 SGD 更新；检查梯度（包括 IndexedSlices）有限且非零，骨干参数实际改变。
5. 学习率类从作者代码 AST 中单独提取；微型测试使用其首步 lr=0.001，总步数采用测试值 200，不声称覆盖完整训练的调度行为。
6. 另外两对图像执行一次 `training=False` 推理，检查输出 2×3、概率有限且每行和为 1，记录真实类别、预测类别与概率。不以两个样本计算正式分类性能。

采用 batch=1 减少显存需求；不自动降低分辨率、不转为小模型、不只训练分类头。若 OOM，保留失败信息，讨论后再改方案。

## 4. Python 命令与运行方式

以下是确认后由 `slurm/smoke.sbatch` 执行的核心命令，**本轮没有执行**。`$RUN` 是本次作业独立目录，`$RUN/code` 是批准提交的 Git 归档副本。缓存、库路径、2 线程限制和禁用 pyc 均由 sbatch 设置。

```bash
# Geoloc，工作目录为 $RUN/code
/scratch/liuzeyu/anaconda3/envs/cvdisaster_geoloc/bin/python -B -u \
  scripts/smoke/geoloc.py \
  --data-root /scratch/liuzeyu/heyinxi/dataset/m1749324/CVIAN \
  --output "$RUN/output"

# Est，工作目录为 $RUN/code
/scratch/liuzeyu/anaconda3/envs/cvdisaster_est/bin/python -B -u \
  scripts/smoke/est.py \
  --data-root /scratch/liuzeyu/heyinxi/dataset/m1749324/CVIAN \
  --output "$RUN/output"
```

确认后先按本地修改 → GitHub → 超算拉取同步，记录包含这些新文件的完整提交号；旧的 `4785bbc` 不包含本轮 smoke 代码，不能用于新作业。先仅提交 Geoloc，读完退出状态和 summary 后再提交 Est；前者失败时先报告，不自动继续。

```bash
# 待批准后执行；COMMIT 必须填写届时已同步且包含 smoke 脚本的提交号。
sbatch --job-name=smoke-geoloc --export=ALL,CVDISASTER_COMMIT="$COMMIT" \
  /scratch/liuzeyu/heyinxi/CVDisaster/slurm/smoke.sbatch geoloc

# Geoloc 结果检查完成后，再单独提交：
sbatch --job-name=smoke-est --export=ALL,CVDISASTER_COMMIT="$COMMIT" \
  /scratch/liuzeyu/heyinxi/CVDisaster/slurm/smoke.sbatch est
```

## 5. 资源与预期输出

两次任务各申请 `a100x4 / wanyi / 1 节点 / 1 任务 / 2 CPU / 1 GPU / 16 GiB 主机内存 / 00:30:00`，顺序提交。Python 子进程最长 27 分钟，预留归档和退出记录时间。只用现有环境，不加载 CUDA 12.2 模块，不运行 pip/conda 安装。

- 单卡目标为分区的 A100 40 GB。Geoloc 显存粗估 4–10 GiB，Est 原始分辨率双分支粗估 12–32 GiB；这些是预算估计，未在本次配置实测，实际值会记录。不会自动申请更多 GPU 或更改 batch/分辨率。
- 计算量仅一个训练 batch 和最小评估/推理。通常预计数分钟到十几分钟；此前共享存储上的首次 TensorFlow 导入曾耗时较长，30 分钟内完成不作保证。排队时间不计入作业时限，2 CPU 也不能保证绕过账号配额限制。
- 日志：`/scratch/liuzeyu/heyinxi/logs/smoke-<job-name>-<JobID>.log`。
- 独立输出：`/scratch/liuzeyu/heyinxi/smoke-tests/<module>-<JobID>/`，含只读来源的代码归档、局部缓存、`exit-code.txt` 和 `output/summary.json`；Geoloc 另有少量沙盒图像及 CSV。
- `summary.json` 逐阶段记录 running/passed/failed、耗时、版本、GPU、样本路径与标签、loss、参数更新验证、推理或检索结果、GPU 峰值分配显存。正常 Python 异常写 traceback 并返回非零；硬超时/OOM 可能只留下最后阶段及 Slurm 日志，不伪装通过。
- 全部检查通过才输出 `GEOLOC_SMOKE_PASSED` 或 `EST_SMOKE_PASSED`，且 summary 状态为 passed、退出码为 0。单有 loss 数值或 GPU 识别信息不足以判定整体通过。
- 不调用 fit 或 epoch 循环，不保存模型权重，不写正式数据或正式训练目录；对所读取图像、方位角文件和预训练权重做前后 SHA256 校验，确认内容未变。所有测试写入限制在新建 smoke 目录，拒绝覆盖已有输出。

## 6. 用户已确认的执行约束

1. 按 Geoloc → 检查结果 → Est 顺序执行单 batch、30 分钟 GPU 方案。
2. 先挂起旧完整训练任务 `3780573`、`3780728`，不取消，也不恢复其完整训练。
3. Geoloc 未全部通过时不提交 Est，不重新安装依赖。

权重已在 `.gitignore` 中排除；本轮补充 `.hdf5` 及 pretrained/weights/checkpoints 目录规则，测试产物也不上传 GitHub。
