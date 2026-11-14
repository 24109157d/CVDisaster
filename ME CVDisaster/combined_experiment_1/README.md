# 合并实验1

研究问题：在CVIAN上，当Geo检索错误时，使用实际检索的错误卫星图片，会使Est的灾损分类准确率变化多少？

## 固定划分与独立训练

全部4,121对样本统一使用Geo的 `train_test_split(train_size=0.6, random_state=42)`：训练2,472对，测试1,649对。按卫星与街景相对路径分别排序后形成图片对，与作者Geo的数据准备方法一致。`prepare_splits.py` 核验已有 `train_60.csv` 和 `val_60.csv`，复制为固定清单，记录SHA256、样本ID和类别数量。Geo与Est都读取这份清单；每个阶段验证共享清单和快照的SHA256。

Geo与Est分别从CVUSA和ImageNet初始权重训练。本次训练产物记录配置、清单SHA256、初始权重SHA256、best权重SHA256、环境和代码提交号。Geo复用 `sample4geo` 的模型、Dataset、训练、评价及相似负样本采样；Est直接复用作者目录的 `EstCheckpoints`。

配置位于 `config.json`：

| 参数 | Geo | Est |
|---|---|---|
| 模型 | 共享ConvNeXt Base | 双分支GCViTTiny |
| 初始化 | CVUSA checkpoint | ImageNet GCViT Tiny |
| 训练随机性 | 沿用复现的seed=42及后端设置 | 沿用复现的训练随机性设置 |
| epoch | 10 | 200 |
| 全局batch size | 64 | 12 |
| optimizer | AdamW | SGD，默认momentum=0 |
| 初始学习率 | 0.0005 | 0.01 |
| 预热 | 1个epoch对应39步 | 10步，从0.001开始 |
| 调度 | cosine，按初始39×10=390步 | cosine，206×200=41,200步 |
| weight decay | 0.01 | 默认值 |
| 训练GPU数量 | 2 | 2 |

seed=42固定两项任务共用的数据划分。Est训练入口保持复现代码的随机性设置。

Geo复用作者的Dataset、图像增强、双向InfoNCE、混合精度和相似负样本采样；label smoothing=0.1，梯度值裁剪100，翻转概率0.5、旋转概率0.75，neighbour_select=64、neighbour_range=128。自定义采样可能舍弃部分尾批样本，调度步数保持原计算位置；每轮实际样本ID及调度步数写入训练记录。

记录中的样本ID统一转换为Python `int`，保持完整整数值；JSON序列化在创建目标文件之前完成。

Est使用原始街景512×1024和卫星512×512，`decode_png`、默认双线性resize及0—255像素值。保持两分支特征拼接、Dense 64、LeakyReLU、Dropout 0.2、三分类softmax和交叉熵。训练仅使用正确配对，按固定训练清单顺序读取。

Geo在训练前评价测试集，随后计算训练集相似负样本并执行shuffle；每轮按训练、测试评价、训练集相似度计算、best更新、shuffle的顺序执行，最后一轮也执行shuffle。按测试R@1严格提高保存best，相同分数保留较早轮次，保存 `weights_end.pth` 后重新加载best评价。

Est每轮通过 `validation_data` 评价同一份40%测试清单，使用原有 `EstCheckpoints` 按 `val_accuracy` 严格提高保存 `0.6_bestCV.h5`；相同分数保留较早轮次。每个完整epoch保存latest，其中包含模型权重、SGD状态、已完成轮次及历史best。训练完成后加载best，释放训练缓存，使用两卡 `predict_on_batch` 评价全部测试样本。

Est仅在显式提供 `--resume-from <旧Est输出目录>` 时从完整latest续训，并在新输出目录保存结果。原有检查点代码核对配置、清单SHA256、TensorFlow版本、权重结构和完整轮次；200轮仍是训练总目标。已完成200轮的latest直接进入best评价。Geo使用原有checkpoint_start初始化方式。

## 测试与主要比较

Geo使用全部1,649张测试街景检索全部1,649张测试卫星，输出Top-1；分数相同时取固定候选清单中位置最前的图片，并记录同分数量。

同一Est权重对全部1,649个查询分别执行：

- `paired`：原始查询街景＋正确配对卫星图片。
- `combined`：同一原始查询街景＋Geo实际检索的Top-1卫星图片。

查询的真实灾损标签用于两次分类评价。两次推理都使用两卡MirroredStrategy及 `predict_on_batch`，图片处理、batch size及尾批处理与复现代码相同。正确配对的预测同时与Est训练结束时保存的逐图预测核对。Geo检索正确时，两次概率与分类必须一致，否则停止运行。

主要结果取Geo检索错误的同一组查询，报告 `combined准确率 − paired准确率`，单位为百分点；负值表示使用错误检索卫星后准确率降低。报告同时保存：

- 样本数量、两次准确率、macro-F1、各类指标及confusion matrix。
- `correct_to_incorrect`、`incorrect_to_correct`、两次均正确、两次均错误的数量。
- 输入替换的有害及有益变化占该组查询的比例。
- 10,000次配对bootstrap的95%准确率变化区间及双侧精确McNemar检验。
- 全测试集、Geo检索正确组、错误卫星与查询类别相同组、类别不同组的对应统计。
- Geo Top-1准确率和地点、等级同时正确的Joint Accuracy。

某组没有样本时，数量记为0，其准确率、差值及统计检验记为null。统计区间按查询重采样，连续街景可能相关，因此区间和检验作为本次固定划分、单次训练的探索性结果。单次串联实验用于判断错误卫星输入的影响，后续模型设计仍需独立实验验证。

## 代码入口

- `scripts/submit_pipeline.sh`：提交一个 `combined_experiment_1` 作业，在同一资源分配中依次固化划分、训练Geo、训练Est、执行检索及Est配对比较；提交参数为 `CVDISASTER_COMMIT`。
- `scripts/pipeline.sbatch`：完整串联作业，使用两张GPU、8个CPU、60,000 MiB主机内存，时间上限48小时。任一阶段失败立即停止。
- `prepare_splits.py --output <固定清单目录>`：检查真实图片及seed=42划分，生成共用快照。
- `reuse_geo.py --reference <复用记录JSON> --check-only`：核验已有Geo权重、原训练配置、原划分、源码、日志及提交记录。
- `train_geo.py --splits <固定清单目录> --output <新Geo目录>`：独立训练Geo。
- `train_est.py --splits <同一固定清单目录> --output <新Est目录> [--resume-from <旧Est目录>]`：独立训练Est或显式续训。
- `retrieve.py --splits <固定清单目录> --geo-run <Geo目录> --output <新评价目录>`：完整测试集的实际检索。
- `estimate.py --splits <固定清单目录> --est-run <Est目录> --output <同一评价目录>`：正确配对与检索配对的逐样本比较。
- `verify.py`：使用本地归档的真实清单和Est预测检查划分、统计及模型结构，并与复现代码核对训练、评价、调度、best和续训相关的关键调用。

训练与推理入口要求Slurm计算节点，均使用两张GPU。完整实验使用 `submit_pipeline.sh` 提交一个作业。`scripts/train.sbatch` 和 `scripts/evaluate.sbatch` 保留单阶段运行及显式续训入口；完整串联作业内直接调用Python入口。

完整串联作业保存在 `/scratch/liuzeyu/heyinxi/runs/combined-experiment-1-作业编号/`，包含 `source/`、`splits/`、`geo/`、`est/` 和 `evaluation/`，同一个JobID贯穿全部阶段。

## 已完成Geo的复用

`configs/geo_4301663.json` 记录作业4301663已完成的10轮Geo训练及best权重，包含真实训练清单、权重、源码和日志的SHA256。提交入口显式接受 `--reuse-geo-reference configs/geo_4301663.json` 时，单个作业依次复用Geo、训练Est、检索及评价。

复用阶段核验原作业的固定清单及当前共享清单与记录中的SHA256完全一致，逐字节复制原作业的 `splits/` 和best权重到新作业目录，并记录权重来源。Est读取新作业中复制的这份固定清单，成员和顺序与Geo原训练划分保持一致。原作业的文件保持原状。

评价输出为 `geo_predictions.csv`、`geo.json`、`predictions.csv` 和 `result.json`；其中逐图结果包含两种输入的预测、全部类别概率、检索卫星、真实卫星及Geo正确性。
