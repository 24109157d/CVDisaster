# Geo CVFT 代码运行说明

日期：2026-11-11。方案编号：`geo_cvft_refine`。

## 模型与数据

`geolocalization/model.py` 提供 `GeoModel`。`baseline` 使用共享 ConvNeXt 和全局向量；`geo_cvft_refine` 增加共享 1×1 通道投影、4×16 街景聚合、8×8 卫星聚合、20 次 FP32 Sinkhorn 和街景残差模块。

训练调用 `model(ground, satellite, sample_ids=ids)`，返回两个字典。单视角调用 `model(image, view="ground", sample_ids=ids)` 或 `model(image, view="satellite", sample_ids=ids)`。字典包含 `global_features` 和 `descriptors`；组合模型还包含 `spatial_features` 和 `fusion_weight`，街景另外提供 `transport` 和 `transport_diagnostics`。`fusion_weight` 对每个样本记录同一个模型比例，供 DataParallel 汇集；`fusion_state()` 返回模型共用的标量记录。

空间向量按位置、通道顺序展开。街景源位置为 `row*16+column`，目标位置为 `row*8+column`。搬移采用 `P.transpose(1,2) @ G`。

两个分支分别在 FP32 中使用 `eps=1e-12` 进行 L2 归一化。共享可训练标量 `fusion_logit` 初始化为 0，`alpha=sigmoid(fusion_logit)`。两个缩放系数通过 `exp(0.5*logsigmoid(fusion_logit))` 和 `exp(0.5*logsigmoid(-fusion_logit))` 计算，缩放后拼接并再次进行 L2 归一化。

训练损失读取最终 `descriptors`，使用一个共用 `logit_scale` 计算双向 InfoNCE，label smoothing 为 0.1，reduction 为 mean。CVUSA checkpoint 的原有参数名称集合与形状必须完整匹配，共用 `logit_scale` 从该权重加载。新增 Conv 和 Linear 使用 Xavier uniform，bias 为零。

`geolocalization/data.py` 直接调用作者的 Dataset、transforms，以及根目录 `scripts/prepare_geoloc_data.py` 的 `validate_splits`。训练使用 `[160:350]` 街景裁剪，评估使用 `[160:360]`，图像尺寸分别为 140×768 和 384×384。共享数据与原有 CSV 仅供读取。

## 配置与输出

配置文件为 `configs/geo_cvft_refine.json`。模型通过命令行选择，训练 seed 为 42、43、44。两张 GPU 使用 DataParallel 汇集特征后，在主卡计算全局 64×64 InfoNCE。每轮自定义采样使用 2,432 对图片，共 38 次参数更新；scheduler 预热 39 步，总长度 390 步，完整训练执行 380 次更新。

训练输出目录包含：

- `run.json`：代码提交号、环境版本、GPU、后端设置、初始 logit scale 与融合比例、可训练参数数量、输入数据配置、增强参数和划分 SHA256。
- `train_60.csv`、`val_60.csv`：本次运行使用的清单副本。
- `submitted.sbatch`：实际提交脚本。
- `steps.jsonl`：最终向量的 InfoNCE 损失、共用 logit scale、更新前后融合参数与比例、融合参数的梯度、学习率、GradScaler 和 Sinkhorn 诊断。
- `samples_epoch_XX.json`、`history.json`：每轮采样顺序、优化步数、评价结果、耗时和内存。
- `best.pth`：按测试 R@1 严格提高保存的 checkpoint。
- `last.pth`：训练完成 checkpoint。
- `best_queries.csv`：全部查询的正确候选排名、分数、同分数量、Top-1 ID 和图片路径。
- `best_topk.csv`：Top-K 卫星 ID、经纬度、最终分数、两个分支分数、融合比例和图片路径。
- `best_satellite_features.pt`：完整卫星候选向量、ID、checkpoint SHA256 和融合参数；加载时必须匹配当前 checkpoint 与融合比例。
- `author_evaluation_consistency.json`、`author_evaluation_queries.csv`：baseline best checkpoint 的作者评价一致性结果及逐查询排名。
- `best_metrics.json`、`result.json`：best 重新加载后的完整指标、分支诊断与资源记录。

两个 checkpoint 均保存模型、共用 logit scale、融合参数、optimizer、scheduler、GradScaler、轮次、优化步数、采样顺序及随机生成器状态。训练入口每次独立初始化。运行目录必须位于代码仓库之外。每次评价重新生成候选特征，记录当前模型的 checkpoint 标识和融合比例。

两种模型共用评价函数。最终向量在 FP32 中再次执行 L2 归一化，并使用 FP32 计算完整相似度矩阵。Recall 沿用作者定义：排名等于分数严格大于正确候选的数量。R@1% 使用 16 个候选。Top-K 清单遇到相同分数时，按照卫星数值 ID 升序排列。

## 计算节点验收

`validate.py` 使用真实 CVIAN 图片和已有 CVUSA 权重，核验完整训练 batch 64、评价 batch 128、两张 A100、单视角独立预测、矩阵与特征顺序、均值关系、向量范数、初始化及真实更新后的融合公式、最终向量损失、各模块梯度、参数更新、完整测试集 ID、融合参数及候选缓存保存加载。它为两种模型分别输出验收 JSON 和验收权重。

`geolocalization/author_validation.py` 对同一个 baseline checkpoint，直接调用作者的 `predict` 和 `calculate_scores`，并与共用评价流程核对完整测试集的 ID 顺序、实际 dtype、相似度、逐查询排名和四项 Recall。相似度最大绝对误差必须不超过 `1e-5`，排名与 Recall 必须一致。验收失败时保存诊断记录并终止。baseline 正式训练的 best 重新加载后，以及独立评价时，均执行该检查。

训练、评估和验收入口要求 Slurm 计算节点环境。运行脚本只执行 `CVDISASTER_MODE` 明确指定的一项操作。

下面命令供以后提交时使用。本次代码编写没有执行这些提交命令。

```bash
cd /scratch/liuzeyu/heyinxi/CVDisaster
git pull --ff-only origin main
export CVDISASTER_COMMIT=$(git rev-parse HEAD)

# 真实数据验收。
sbatch --export=ALL,CVDISASTER_MODE=validate "ME CVDisaster/scripts/experiments/geo_cvft_refine.sbatch"

# 每次明确指定一个模型和一个训练 seed。
sbatch --export=ALL,CVDISASTER_MODE=train,CVDISASTER_ARCHITECTURE=baseline,CVDISASTER_SEED=42 "ME CVDisaster/scripts/experiments/geo_cvft_refine.sbatch"
sbatch --export=ALL,CVDISASTER_MODE=train,CVDISASTER_ARCHITECTURE=geo_cvft_refine,CVDISASTER_SEED=42 "ME CVDisaster/scripts/experiments/geo_cvft_refine.sbatch"

# 独立评价已有实验权重。
export CVDISASTER_CHECKPOINT=/scratch/liuzeyu/heyinxi/runs/<模型-seed-JobID>/output/best.pth
sbatch --export=ALL,CVDISASTER_MODE=evaluate "ME CVDisaster/scripts/experiments/geo_cvft_refine.sbatch"
```

另外两组正式比较使用相同命令分别指定 seed 43 和 44。提交前使用集群允许的资源查询命令确认账号和分区可用性。

完成六次训练后，`compare.py --runs <六个output目录> --output <仓库外的新目录>` 读取各次 `result.json`、`run.json` 和 `best_queries.csv`，核验模型、seed、公共配置、划分、测试 ID、逐查询指标和 baseline checkpoint 的作者评价一致性，输出均值、样本标准差、三个成对差值、融合比例与四种正确性组合的数量。

## 参考代码

- CVFT：`YujiaoShi/cross_view_localization_CVFT`，提交 `ddf6c2f6d35c0103b9315d686a3135a1a09db5fd`，已阅读完整 README 和 `script/ot_net.py`。
- DenseFlow：`XianghuiZe/DenseFlow_localization`，提交 `16d9f106e9cea4c04458e4cfb9be35f7fba5ece3`，已阅读完整 README，以及 `models_vigor.py`、`gen_BEV/VGG.py` 和 `op_flow/raft.py` 的相关实现。
- 本实验各层尺寸、激活函数、初始化、温度和损失配置依据仓库内实验方案。
