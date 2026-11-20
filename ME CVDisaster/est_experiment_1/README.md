# EST experiment 1

## 模型与输入

当前模型为 ConvNeXt-Tiny＋新融合，入口名称为 `full`。输入为当前街景、正确配对卫星和最多两张附近街景，输出当前街景的 Minor、Moderate、Severe 灾损等级。

- 当前街景与邻图尺寸为 512×1024，共享一个 ConvNeXt-Tiny 和特征投影层。
- 卫星尺寸为 512×512，另一个 ConvNeXt-Tiny 处理卫星。全幅图像和中心 256×256 裁切后放大的图像共享卫星编码器。
- 两个编码器均读取 Keras 官方 ImageNet-1K `convnext_tiny_notop.h5`。读取前检查配置中的 SHA256。图像像素范围为 0—255，ImageNet 标准化包含在 ConvNeXt 内部。
- 双尺度只使用已有卫星覆盖范围，中心裁切没有增加可观察的地理范围。两个尺度分别具有空间位置参数。

## 融合计算

地面特征保持 16×32 个空间位置，每个位置投影为 256 维。每个卫星尺度平均池化为 8×8 个位置，共 128 个 Key/Value Tokens。街景 Query 使用两个 Attention Heads，总投影维度为 128，每个 Head 为 64 维。

每个街景位置读取两个尺度的卫星特征，生成通道调制参数。FiLM 使用有界残差：`F + 0.1 * (tanh(gamma) * LayerNorm(F) + tanh(beta))`。生成 gamma、beta 的最后一层采用零初始化，初始调制保持原特征。

邻图经同一卫星调制后，每张平均池化为 8×16 个位置。目标的每个位置分别读取每张邻图；Sparsemax 直接作用于 Patch 相关性分数。每张邻图具有一个可学习 Null Key，其 Value 为零。缺少邻图时，mask 仅允许 Null。附近图像是否具有相同标签不参与注意力计算。

读取结果根据相似度阈值 0.8 计算重复程度，分配累计权重，汇总为邻图增量。相同图片及相同关系重复出现时共同分配权重。该机制不承诺排除全部无关内容。

Base 使用当前街景及卫星调制特征。Base 与 Final 共享 `Dense(64) → LeakyReLU → Dropout(0.2) → Dense(3, softmax)` 分类 Head。Base 的归一化预测熵经过 `tf.stop_gradient`；门控随该熵单调增加，控制邻图残差。

门控前两个 epochs 的释放系数为零，第 2—5 个 epochs 线性增加至 1。释放系数与 Sigmoid 置信度门控相乘。模型权重保存训练进度，恢复时从 optimizer steps 继续计算进度。推理时邻图增量为零，Base 与 Final 概率一致。

## 训练参数

- SGD，momentum=0；学习率 0.01，从 0.001 预热 10 个 optimizer steps，随后使用以 200 epochs 为总目标的 cosine 调度。
- 有效 batch size 为 12，每个设备逐样本累计梯度，最后一个 batch 保留全部样本。
- 训练随机种子为 42、626、2024、2025、2026，五次训练使用同一份数据清单。外部划分种子与内部验证划分种子均为 626。
- 联合损失为 `CE(Final) + 0.3 CE(Base) + 0.2 Lord(Final) + 0.1 CE(Satellite)`。
- `Lord = 0.5 * sum_k w[k] * (sum_{c<=k} p[c] - 1[y<=k])²`，k 为 0、1，默认分界权重为 `[1,1]`。权重只允许依据独立验证集合选择。
- 邻图要求同一 Sequence、距离不超过 50 米、时间间隔不超过 60 秒，最多两张，允许时间之前与之后的观察。
- 训练只使用正确配对卫星。

固定划分直接读取复现 EST 作业 `4087311` 的 `est_train_60.csv`、`est_test_60.csv` 和 `est_config.json`。该作业采用 `SEED=626`，通过 TensorFlow 对排序后的街景路径执行一次 shuffle，再按 6:4 执行 take/skip。程序检查实际清单 SHA256，保持 2,472 个训练目标与 1,649 个测试目标的内容、类别及顺序。

训练集合内部采用 `GroupShuffleSplit(test_size=0.2, random_state=626)`，按 Sequence 生成 1,971 个训练目标及 501 个独立验证目标，两者的 Sequence 互斥。使用验证 accuracy 选择 epoch，从 ImageNet 重新初始化，在全部 2,472 个训练目标上训练选定轮次，最后评价测试集合。测试集合不参与参数选择。

各集合分别选择自身集合内的邻图。外部随机划分允许完整训练与测试具有相同 Sequence，报告结果时需要说明。训练及评价入口同时检查划分协议、种子、比例、参考清单指纹与邻图规则。

## 评价与产物

`evaluate.py` 固定同一 checkpoint，对全部测试目标执行正确配对、五份随机错误卫星、五组均匀噪声、五组 Gaussian Noise、全零卫星评价。可读取已有 Geo 检索清单。每种条件均重新计算卫星调制及 Base/Final，评价期间模型参数不更新。

Gaussian Noise 使用固定随机种子，裁切前均值 127.5、标准差 `sqrt(5461.25)`，裁切至 0—255 后使用 float32；均匀噪声保持既有 uint8 生成规则。Gaussian 指纹基于 little-endian float32，均匀噪声指纹基于 uint8。全零卫星表示标准化前像素全部为零。

每种条件输出 OA、Macro-F1、各类别 precision/recall/F1、混淆矩阵和 NLL。CSV 同时保存 Base/Final 概率与预测。每种条件及每个类别输出：

- `Nfixed`：Base 错误、Final 正确。
- `Ncorrupted`：Base 正确、Final 错误。
- `delta_N = Nfixed - Ncorrupted`。
- `net_benefit_pp = 100 * delta_N / N`。

`analysis.py` 使用配对 Sequence 或空间分组，输出差值、95% bootstrap 区间、分组置换检验和同族 Holm 校正。默认比较包含正确卫星下的 Final/Base 净收益，以及正确卫星分别相对四种替换条件的差值。五个训练种子的相同目标不会作为五倍独立目标。

卫星替换差值衡量同权重模型的输入敏感性。模型性能提升与卫星贡献的论文结论，还需要使用统一数据、训练和评价规则的相应对照结果。本实验当前只提供所选模型的代码。

## 工作空间

- `config.json`：输入尺寸、预训练权重路径及全部训练参数。
- `models.py`：编码器、新融合、共享分类 Head、损失及调度。
- `prepare.py`、`dataset.py`、`conditions.py`：固定划分、邻图和评价输入。
- `train.py`、`training.py`、`pipeline.py`：验证选择、重训、续训和多种子执行。
- `evaluate.py`、`analysis.py`：完整评价与配对统计。
- `validation_cases.py`、`verify_model.py`：真实训练图片的模型检查入口。
- `tests/`：真实元数据和历史预测的检查。
- `docs/运行说明.md`、`docs/本机检查记录.md`：命令与验证范围。
- `scripts/`：资产准备、双 GPU 模型检查、五种子训练数组及统一统计定义，提交作业需要用户授权。
- `.local-cache/`：固定本机清单、历史诊断、依赖及测试缓存，已由 Git 忽略。

## 实现依据

[Keras 2.10 ConvNeXt 源码](https://raw.githubusercontent.com/keras-team/keras/v2.10.0/keras/applications/convnext.py)提供编码器、预处理及官方权重 SHA256；[TensorFlow Addons Sparsemax](https://raw.githubusercontent.com/tensorflow/addons/v0.18.0/tensorflow_addons/activations/sparsemax.py)提供稀疏注意力激活。
