# CVDisaster Geo 特征映射与空间修正实验方案及运行说明

日期：2026-11-12。

方案编号：`geo_cvft_refine`。

状态：用户已确认当前模型和原 geo 训练流程，并授权在检查超算环境后提交一次 `geo_cvft_refine`、seed=42 的正式训练。

本次任务和参数见 [2026-11-12 正式训练提交记录](Geo_CVFT_seed42_2026-11-12_正式训练提交记录.md)。已有提交情况见 [作业提交记录](Geo_CVFT_2026-11-11_提交记录.md)，各记录描述对应时间的作业状态。

## 1 实验问题与比较对象

研究共享 ConvNeXt 提取的空间特征经过 CVFT 搬移和残差修正后，能否提高 CVIAN 街景到卫星图像的检索准确率。

比较对象固定为两种网络：

- `baseline`：当前已经复现的 CVDisaster Geo，使用共享 ConvNeXt 和全局检索向量。
- `geo_cvft_refine`：共享 ConvNeXt，同时输出全局特征与经过 CVFT 搬移、残差修正的空间特征，通过可学习比例组合成最终检索向量，使用该向量的 InfoNCE 共同训练。

本实验评价完整组合结构的效果。结论归属于整体网络，单个模块的贡献需要相应证据支持。

基线直接使用已完成的 geo 复现结果，来自作业 `3979539`，训练提交为 `8ca60d598d2c306930b1b61d80c098ece846efeb`，训练 seed 为 42。保存的日志给出 R@1 为 79.8059%、R@5 为 97.8169%、R@10 为 98.9691%、R@1% 为 99.5149%。这些数值作为本次直接比较依据。

本次训练对象仅为 `geo_cvft_refine`，使用 seed=42 完成一次正式训练。比较时使用上述基线日志、已保存的 checkpoint 和已有预测记录。

本地方案编写时的仓库提交为 `9f495dd79854026a6449e21d34a3d04cfc18d3f4`。未来实验记录填写实际训练代码提交号。

## 2 数据与监督信息

### 2.1 数据位置和内容

服务器原始数据目录为 `/scratch/liuzeyu/heyinxi/dataset/m1749324/CVIAN/`。

- 街景：`00_SVI/`，共 4,121 幅。
- 已有方向处理结果：`00_SVI_aligned/`。
- 卫星切图：`01_Satellite/`，共 4,121 幅。
- 位置记录：`02_Position/CVIAN_position.geojson`。

正确配对由图片 ID 确定。类别目录中的损伤标签用于数据完整性核验，geo 的损失函数只使用图片 ID 配对关系。

### 2.2 固定数据划分

两种网络共用已有 `train_60.csv` 和 `val_60.csv`：

- 训练：2,472 对唯一图片。
- 测试：1,649 对唯一图片。
- 划分方式：作者排序后的配对清单，使用 `train_test_split(train_size=0.6, random_state=42)`。
- 划分随机种子始终为 42，与训练随机种子分开记录。

核验两份清单的文件 SHA256、唯一 ID 数量、配对图片名称、类别目录、文件存在性，以及训练和测试之间的 ID 交集。清单及已有方向处理图像保持原样。

本实验沿用作者的训练和测试两部分评价协议。测试集同时用于每轮评价和最佳 checkpoint 选择，结果应按这一协议解释。

### 2.3 数据能够提供的监督

街景和卫星的图片配对监督网络学习检索特征。搬移矩阵通过检索损失学习，其空间位置表示网络内部的特征布局。矩阵中的数值不能直接解释成具有真实地面尺度的像素对应概率。

本结构的前向计算使用图像内容与图像内的空间顺序。相机高度、深度、逐像素对应和物体分割标签均不作为输入。卫星经纬度在检索出图片 ID 后用于输出位置。

## 3 图像输入与预处理

### 3.1 输入尺寸

沿用当前 geo 的图像处理过程，保证两种网络使用相同输入。

- 训练街景：读取 `00_SVI_aligned` 图像，保留行区间 `[160:350]`，随后缩放到高 140、宽 768。
- 评估街景：读取相同方向处理结果，保留 `[160:360]`，随后缩放到高 140、宽 768。
- 卫星：读取完整切图，缩放到高 384、宽 384。
- 图像通道顺序：RGB。
- 缩放插值：沿用 `cv2.INTER_LINEAR_EXACT`。
- 归一化：使用现有模型的 `get_config` 返回值，并保存实际 mean、std 和数据配置。

训练与评估街景裁剪区间的上述差别来自当前基线，本次结构比较沿用这一处理规则。

### 3.2 训练增强

增强顺序及概率沿用当前 Dataset 与 transforms：

1. 图片对以 0.5 的概率同时水平翻转。
2. 各视角以 0.5 的概率进行质量 90 至 100 的图像压缩增强。
3. 缩放到对应网络输入尺寸。
4. 各视角以 0.5 的概率执行 ColorJitter，brightness、contrast、saturation、hue 参数均为 0.15。
5. 各视角以 0.3 的概率执行 AdvancedBlur 或 Sharpen。
6. 各视角以 0.3 的概率执行 GridDropout 或 CoarseDropout。卫星 GridDropout 的 ratio 为 0.4，街景为 0.5；CoarseDropout 的具体参数沿用原 transforms，并记录配置。
7. 归一化并转换为 tensor。
8. 图片对以 0.75 的概率执行对应旋转：卫星旋转 90、180 或 270 度，街景在水平方向进行对应的周期移动。

颜色、压缩和遮挡增强分别采样；图片对的翻转与方向变化保持关联。评估使用确定性的裁剪、缩放和归一化。

## 4 网络结构

### 4.1 共享编码器与两个输出分支

backbone 固定为 `convnext_base.fb_in22k_ft_in1k_384`。街景与卫星调用同一个 ConvNeXt 实例，共享全部 backbone 参数。

通过 `forward_features` 取得最后一个 stage 的空间特征，再通过原有 `forward_head` 取得全局向量。同一次图像编码产生的空间特征同时服务于全局分支和空间分支。

```mermaid
flowchart TD
    G[街景 140×768] --> B[共享 ConvNeXt Base]
    S[卫星 384×384] --> B
    B --> U[原有全局分支 输出各自的 1024 维向量]
    B --> H[共享通道投影 1024 到 64]
    H --> GH[街景聚合为 4×16]
    H --> SH[卫星聚合为 8×8]
    GH --> P[根据街景生成 CVFT 搬移矩阵]
    P --> T[搬移街景特征 排列为 8×8×64]
    T --> R[街景空间特征残差修正]
    R --> VG[展开并归一化 街景空间向量 4096 维]
    SH --> VS[展开并归一化 卫星空间向量 4096 维]
    U --> Z[按可学习比例缩放并拼接 各自输出 5120 维]
    VG --> Z
    VS --> Z
```

### 4.2 空间 tensor 的形状

使用 `B` 表示图片对数量，`d=64` 表示空间分支的通道数量。

根据当前输入尺寸和 ConvNeXt 的下采样规则，预期空间特征为：

- 街景 `F_g`：`[B, 1024, 4, 24]`。
- 卫星 `F_s`：`[B, 1024, 12, 12]`。

这两项形状在实现验收时使用真实 CVIAN 图片核对。

共享 `Conv2d(1024, 64, kernel_size=1, bias=True)` 完成通道投影。随后分别使用 `AdaptiveAvgPool2d((4,16))` 和 `AdaptiveAvgPool2d((8,8))` 聚合位置。按行优先顺序展开为：

\[
G\in\mathbb{R}^{B\times64\times64},\qquad
S\in\mathbb{R}^{B\times64\times64}.
\]

第二维为空间位置，第三维为特征通道。街景源位置索引采用 `i=row*16+column`，卫星目标位置索引采用 `j=row*8+column`。

### 4.3 全局分支

全局分支沿用现有 `forward_head`，保持原有 pooling 与归一化层的顺序，输出 `[B,1024]`。按第 7 节的规则进行 L2 归一化，得到 `u_g` 和 `u_s`。

共享编码器接收两个分支的训练梯度。因此，组合模型训练后的全局特征也可能改变，其评价结果需要由训练后的 checkpoint 计算。

## 5 CVFT 特征搬移

### 5.1 代价矩阵生成

矩阵生成模块只接收街景的 64 通道、4×16 空间特征：

1. 使用 `Conv2d(64,1,kernel_size=1,bias=True)` 压缩通道。
2. 使用 ReLU。
3. 按街景源位置顺序展开为 `[B,64]`。
4. 使用 `Linear(64,4096,bias=True)` 生成矩阵数值。
5. reshape 为代价矩阵 `C(g)`，形状为 `[B,64,64]`。

矩阵的行对应街景源位置，列对应目标网格位置。每幅街景独立生成自己的矩阵；同一矩阵应用于该图像的全部 64 个通道。

初始化参考 CVFT 的 `script/ot_net.py`。共享通道投影和通道压缩 Conv 的 weight 使用 Xavier uniform；bias 也采用一维 Xavier uniform，输出通道数为 `n` 时使用区间 `[-sqrt(3/n),sqrt(3/n)]`。矩阵生成 Linear 的 weight 使用均值 0、标准差参数 0.005 的截断正态分布，截断区间为 `[-0.01,0.01]`；bias 为按行展开的 `I_64`。这个单位矩阵仅定义代价值的初始参数，经过负号、行列归一化后生成实际传输矩阵；源网格与目标网格的相同展开序号不代表已知的地理对应关系。原预训练模型的 backbone 和全局 head 参数按 checkpoint 原值加载。

### 5.2 Sinkhorn 计算

主配置固定：

- 温度 `transport_tau=1.0`。
- 行列归一化迭代次数固定为 `sinkhorn_iterations=30`，训练和评估使用相同次数。
- 计算 dtype 为 FP32；在 AMP 训练中，该模块显式使用 FP32。
- 在对数域计算，初值 `log_P=-C(g)/transport_tau`。

CVFT 原作者使用 `log_P=-100*C(g)` 和默认 20 次迭代。本实验使用已确认的 `transport_tau=1.0` 和固定 30 次，代价值的缩放属于本实验配置，按该配置记录数值范围和检索结果。

每次迭代依次执行：

\[
\log P_{ij}\leftarrow\log P_{ij}
-\operatorname{logsumexp}_{k}(\log P_{ik}),
\]

\[
\log P_{ij}\leftarrow\log P_{ij}
-\operatorname{logsumexp}_{k}(\log P_{kj}).
\]

最后通过指数运算得到 `P`。这里采用各行和各列之和均约为 1 的双随机矩阵约定，矩阵总和约为 64。

每步核验矩阵的有限性、非负性和行列和误差。定义：

\[
e_{row}=\max_i\left|\sum_jP_{ij}-1\right|,
\qquad
e_{col}=\max_j\left|\sum_iP_{ij}-1\right|.
\]

行列和误差仅作为诊断数据记录，不设置误差阈值，不根据误差大小提前结束迭代或终止训练。每次前向固定执行 30 次行列归一化。代价矩阵、传输矩阵或行列和误差出现非有限值，以及传输矩阵出现负值时，就地报错，输出样本 ID、代价范围、行列误差及温度。迭代次数与温度的更改需要更新配置和方案，正式运行不自动改变计算规则。固定 30 次属于近似计算，不保证行列和达到某个指定精度，其数值误差和检索效果通过实验记录评价。

### 5.3 特征搬移

对于每个样本，使用：

\[
\widetilde G=P^\top G.
\]

因此：

\[
\widetilde G_j=\sum_iP_{ij}G_i.
\]

`P` 的行列约定与上述乘法顺序保持一致。搬移输出形状为 `[B,64,64]`，按目标位置索引恢复为 `[B,64,8,8]`，交给空间特征修正模块。

最终卫星图片及其正确 ID 不参与预测时的街景矩阵生成。跨视角关系通过训练损失传入矩阵生成模块。

### 5.4 空间顺序的保留

对双随机矩阵有：

\[
\operatorname{mean}(P^\top G)=\operatorname{mean}(G).
\]

所以本结构保留目标网格顺序，最终展开空间特征。4×16 和 8×8 的自适应聚合发生在搬移之前，负责形成有限的位置集合。

记录每轮矩阵的平均行熵、最大元素、`e_row` 和 `e_col`。行熵为：

\[
H(P)=-\frac1{64}\sum_i\sum_jP_{ij}\log(P_{ij}+10^{-12}).
\]

接近均匀的矩阵可能过度混合位置，过度集中的矩阵也需要结合损失与测试表现判断。熵仅作为诊断量，主损失中不增加对应正则项。

## 6 空间特征修正

修正模块接收搬移后的街景特征 `X`，形状为 `[B,64,8,8]`，输出同形状 tensor：

\[
Y_1=\operatorname{ReLU}\left(\operatorname{IN}_1
\left(\operatorname{Conv}_{3\times3}^{(1)}(X)\right)\right),
\qquad
Y_2=\operatorname{ReLU}\left(\operatorname{IN}_2
\left(\operatorname{Conv}_{3\times3}^{(2)}(Y_1)\right)\right),
\qquad R(X)=\operatorname{ReLU}(X+Y_2).
\]

模块参数固定为：

- 每次卷积后使用 InstanceNorm2d，对每个样本的每个通道按 8×8 空间范围归一化，`eps=1e-5`、`affine=False`、`track_running_stats=False`，训练与评价均使用当前输入的统计量。
- 两个 Conv 均为 64 输入通道、64 输出通道、kernel size 3、stride 1、padding 1、`bias=True`。
- 每个 Conv 后依次执行 InstanceNorm 和 ReLU，残差相加后再次执行 ReLU。
- Conv weight 使用 Kaiming normal，`mode=fan_out`、`nonlinearity=relu`；Conv bias 保留 PyTorch Conv2d 的默认初始化，与 DenseFlow 原修正网络的初始化方式一致。

该计算顺序参考 DenseFlow 的 `gen_BEV/VGG.py::ResidualBlock`。`op_flow/raft.py` 使用 `RefineNet(norm_fn='instance')`，`RefineNet` 对卷积执行上述 Kaiming 初始化。本实验将其中一个等尺寸残差单元用于 64 通道、8×8 特征，模块深度和通道数按当前检索结构设置。

该模块只作用于经过搬移的街景特征。卫星空间特征采用共享通道投影和 8×8 聚合后的输出。

CVFT 负责重新分配空间位置的特征，残差模块在目标网格上学习邻近位置和通道之间的进一步组合。修正输出在这一阶段仍然是 64 通道、8×8 的检索特征，其地理解释需要额外证据支持。

将 `R(X)` 与卫星空间特征分别按相同目标顺序展开为 `[B,4096]`，按第 7 节的规则进行 L2 归一化，得到 `v_g` 和 `v_s`。

## 7 最终输出与相似度

### 7.1 分支归一化与可学习比例

全局分支和空间分支的原始输出分别记为 `h` 和 `r`。每幅图像独立进行：

\[
u=\frac{h}{\|h\|_2},\qquad
v=\frac{r}{\|r\|_2}.
\]

范数沿特征维度计算，输出形状分别为 `[B,1024]` 和 `[B,4096]`。分支归一化采用 FP32，核验原始向量的有限性及非零范数，再使用 `eps=1e-12` 的 L2 归一化函数。图片之间、分支之间均不合并计算范数。

设置一个整个模型共用的可训练标量 `fusion_logit=a`：

\[
\alpha=\operatorname{sigmoid}(a),\qquad
w_{global}=\alpha,\quad w_{spatial}=1-\alpha.
\]

初始化 `a=0`，对应初始比例各为 0.5。街景、卫星、所有样本和所有候选共用这个比例。`a` 通过训练集最终检索损失的梯度更新，预测时使用所加载 checkpoint 内的数值。

### 7.2 缩放与拼接

对归一化分支使用权重平方根进行缩放：

\[
z_g=[\sqrt{\alpha}\,u_g;\sqrt{1-\alpha}\,v_g],\qquad
z_s=[\sqrt{\alpha}\,u_s;\sqrt{1-\alpha}\,v_s].
\]

最终向量维度为 5,120。由两个分支各自的单位范数得到：

\[
\|z\|_2^2=\alpha\|u\|_2^2+(1-\alpha)\|v\|_2^2=1.
\]

街景与卫星的相似度恰好为：

\[
s(g,s)=z_g^\top z_s
=\alpha u_g^\top u_s+(1-\alpha)v_g^\top v_s.
\]

平方根系数保证点积中的比例等于学习得到的 `alpha`。每个分支单独归一化，使维度数量与融合比例分别控制。训练、难负采样和预测使用同一拼接公式。

FP32 中通过 `exp(0.5*logsigmoid(a))` 与 `exp(0.5*logsigmoid(-a))` 计算两个缩放系数，保持数值计算与上述公式一致。融合参数使用与其他可训练参数相同的 optimizer 和参数衰减规则。记录 `a`、`alpha` 及其梯度，检查非有限值。

网络输出两个分支按平方根系数缩放后的拼接向量。训练时由原复现的 InfoNCE 执行最终 L2 归一化，评价时由原复现的 predict 执行最终 L2 归一化，保持各入口原有的归一化和 AMP 执行顺序。baseline 接口直接提供原有全局 head 输出。5,120 维 float32 向量组成的 1,649 张卫星候选库约占 32.2 MiB，此值只表示向量数据本身的存储量。

### 7.3 训练接口

训练样本输入为街景 tensor、卫星 tensor 和图片 ID。网络提供：

- `global_features`：各视角 `[B,1024]`。
- `spatial_features`：各视角 `[B,4096]`。
- `descriptors`：各视角 `[B,5120]`。
- `transport`：街景 `[B,64,64]`，用于搬移和诊断。
- `fusion_weight`：模型共用的标量 `alpha`，用于记录融合比例。

图片 ID 用于配对、预测结果关联和模块诊断，不作为模型的图像特征输入。

### 7.4 单幅图像预测接口

预测显式指定视角 `ground` 或 `satellite`。街景执行矩阵生成、搬移和修正，卫星执行通道投影及目标网格聚合。每幅图像独立输出一个 `[5120]` 检索向量。

测试时对 1,649 张卫星图像预先计算向量，街景向量与整个候选库计算相似度并排序。街景和卫星特征由同一 checkpoint 产生，使用其中同一个 `alpha`。每次评价重新计算向量，并记录 checkpoint 标识与 `alpha`。

逐查询文件记录 Top-K 卫星 ID 和最终相似度、正确候选及 Top-1 候选的分支相似度，融合比例记录在结果 JSON 中。原始图片路径同时保留，供后续实验关联。

## 8 训练损失与温度参数

直接调用原复现的 `sample4geo.loss.InfoNCE`。适配类 `DescriptorInfoNCE` 只从模型输出字典中取出 `descriptors`，归一化、logit scale 的乘法顺序、矩阵乘法和双向 cross entropy 均由原函数执行。原复现的 `sample4geo.trainer.train` 在同一 autocast 范围内执行模型前向和损失计算，再依次执行 backward、unscale、梯度裁剪、GradScaler 更新、清理梯度及 scheduler 更新。新增模块的前向适配和记录回调不改变上述计算顺序。

对于全局 batch `B=64`，将传入 InfoNCE 的向量按其原有代码归一化后，令：

\[
Z_g,Z_s\in\mathbb{R}^{B\times5120},\qquad
y_i=i,\quad i=0,\ldots,B-1.
\]

由一个共用的可训练 `logit_scale=ell` 生成 logits：

\[
M=\exp(\ell)Z_gZ_s^\top\in\mathbb{R}^{B\times B}.
\]

唯一训练目标为：

\[
L=\tfrac12\left[
\operatorname{CE}_{0.1}(M,y)
+\operatorname{CE}_{0.1}(M^\top,y)
\right].
\]

`CE_0.1` 为 label smoothing 0.1、`reduction=mean` 的 cross entropy。正样本为当前行对应的同 ID 卫星，其他 63 幅卫星作为 batch 内负候选；转置方向使用相同定义。baseline 将其最终 1,024 维向量传入同一损失函数。

`logit_scale` 从同一 CVUSA checkpoint 加载并参与训练，与作者原有参数含义一致。Sinkhorn 的 `transport_tau`、InfoNCE 的 `logit_scale` 和融合的 `fusion_logit` 各自承担不同作用，分别记录。

最终融合向量的梯度共同更新共享 ConvNeXt、原有全局 head、共享通道投影、矩阵生成模块、残差修正模块、`fusion_logit` 和 `logit_scale`。两个分支的单独相似度仅作诊断；总损失不附加分支独立损失。

两张 GPU 的最终融合向量汇集后计算 `[64,64]` logits，保持跨 GPU 的梯度连接。所有副本使用同一个逻辑上的 `fusion_logit` 和 `logit_scale`。梯度累积不能代替同一次 InfoNCE 中的负候选数量。

## 9 训练配置

### 9.1 环境和输入权重

- Conda 环境：`/scratch/liuzeyu/anaconda3/envs/cvdisaster_geoloc`。
- Python：3.10，实际补丁版本写入实验记录。
- PyTorch：`1.13.1+cu116`。
- torchvision：`0.14.1+cu116`。
- timm：`0.8.19.dev0`。
- 其他依赖沿用仓库的 `requirements.txt` 与 `requirements-geoloc.lock.txt`，运行时保存实际版本清单。
- 初始 CVUSA 权重：`/scratch/liuzeyu/heyinxi/pretrained/weights_e40_98.6830.pth`。
- 权重 SHA256：`739cf00d303836fba1195f1bb8d54c4b13c53926ce3212fc9ceced335fc59a23`。

组合模型使用与已完成 geo 复现相同的初始 CVUSA checkpoint，沿用 `load_state_dict(..., strict=False)` 加载原有 backbone、全局 head 和 `logit_scale`；新增模块按本方案初始化，`fusion_logit` 初始化为零。运行环境使用已有本地权重创建模型，保持离线运行。

### 9.2 优化参数

- GPU：两张 A100，保持当前全局 batch 设置。
- 全局训练 batch：64 对，每张 GPU 约 32 对。
- 全局评估 batch：128 幅。
- 训练轮数：10。
- 优化器：AdamW。
- 学习率：`5e-4`，backbone 和新增模块使用相同基础学习率。
- AdamW betas：`(0.9,0.999)`，eps：`1e-8`，weight decay：`0.01`，amsgrad：`False`。
- 参数衰减采用当前代码规则，不额外排除特定参数。
- AMP：沿用原复现的 autocast 范围与算子精度选择；GradScaler 初始 scale 为 `2**10`。InfoNCE 直接在原 train 函数的 autocast 范围内执行。Sinkhorn 及组合向量内部的 FP32 计算属于新增模块的适配。
- 组合模型训练目标：最终融合向量的双向 InfoNCE；融合比例通过训练集反向传播学习。
- `fusion_logit` 初值为 0；其参数更新使用同一基础学习率与 weight decay。
- 梯度处理：AMP unscale 后，沿用逐元素梯度裁剪 `clip_grad_value_=100`。
- 正式训练的梯度缺失、零值和非有限值仅记录，不增加梯度终止条件。非有限梯度由 GradScaler 按原复现流程处理；记录回调在参数更新后、清理梯度前收集裁剪后的梯度幅度、缺失参数名称和非有限梯度参数名称。
- scheduler：cosine，`warmup_epochs=1`，按原复现首次采样前的 DataLoader 长度计算预热步数和总调度步数。
- DataLoader workers：4。
- 本次正式训练独立初始化并保存输出，不自动续训。

### 9.3 自定义采样与调度计数

沿用当前训练集自定义采样，`custom_sampling=True`、`sim_sample=True`、`gps_sample=False`、`neighbour_select=64`、`neighbour_range=128`。

调度步数在第一次自定义采样之前确定：`warmup_steps=len(train_loader)*warmup_epochs`，`training_steps=len(train_loader)*epochs`。当前 2,472 对数据和 batch 64 对应 39 批，因此计算出的预热长度为 39 步、总调度长度为 390 步，实际计算值写入 `run.json`。

每轮直接使用原 `Dataset.shuffle` 返回的样本顺序和数量，DataLoader 按实际长度执行训练，逐步调用 scheduler。记录实际样本清单、batch 数量、优化步数和学习率，训练按配置完成 10 个 epoch。

沿用原复现顺序：训练前先评价测试集，再提取训练集特征、计算训练集 Recall 和难负候选，然后执行首次 shuffle；每轮训练结束后依次评价测试集、计算训练集 Recall 和难负候选、选择 best，再执行 shuffle，包含最后一轮。训练集 Recall 调用原 `calculate_scores`，难负候选调用原 `calculate_nearest`。特征和样本 ID 保留在 GPU，使用原有 GPU 矩阵乘法计算分数，再沿用原函数的 CPU 排序与候选筛选。模型产生的实际难负候选可能不同，每轮样本顺序统一记录在 `history.json` 中。

### 9.4 随机种子

数据划分和本次训练均只使用 seed=42。改进模型正式训练次数为一次，训练轮数为 10。

沿用已完成 geo 复现的 `setup_system(seed=42)`，设置 Python、NumPy、PyTorch 和 CUDA 的全局随机种子。DataLoader 沿用原复现的构造参数和默认 worker 随机初始化，不额外传入 `generator` 或 `worker_init_fn`。运行记录保存 seed、GPU、CUDA、cuDNN 和实际后端设置。

CVFT 的模型初始化代码没有为矩阵生成模块单独设置随机种子；`input_data_cvusa.py` 中的 `random.seed(2019)` 位于测试图片读取函数。DenseFlow 的训练入口设置 `np.random.seed(2022)`，其修正模块构造函数没有独立设置 PyTorch seed。本实验通过 `module_init_seed=42` 控制新增模块初始化，初始化分布按第 5、6 节执行。该参数可独立配置，并随完整配置写入运行记录和 checkpoint。

数据划分直接读取已有的 `train_60.csv` 和 `val_60.csv`，记录清单 SHA256。相同清单保证划分一致。新增通道投影、矩阵生成和空间修正模块在 CPU 随机状态保存与恢复的范围内创建，使用 `torch.default_generator.manual_seed(module_init_seed)`，创建完成后恢复原有 CPU 随机状态，CUDA、Python 和 NumPy 的随机状态保持原值。新增模块的初始化不消耗原有训练流程后续使用的随机数。共享 backbone 仍由两条分支共同训练，模型输出和难负候选可能变化，相同随机设置不能保证整个训练过程使用完全相同的增强序列，也不能保证所有 GPU 算子逐位一致。

### 9.5 与作者实验的比较范围

直接改进幅度由本次组合模型的结果与已完成 geo 复现结果计算。改进模型沿用该次复现的初始输入权重、划分清单、预处理、增强规则、全局 batch、GPU 数量、优化器、学习率、训练轮数、调度计数、评价频率、seed=42 及最佳权重选择规则。损失函数沿用作者的双向 InfoNCE 和 label smoothing 0.1。

已完成 geo 复现的配置为全局 batch 64、LR `5e-4`、随机 6:4。作者公开训练入口默认全局 batch 128、LR `1e-3`，默认划分为 `subarea5`；项目记录的正式论文正文 LR 为 `1e-4`。实验报告分别列出作者论文、公开代码及本次沿用的复现设置。论文中的随机 6:4 表格提供背景参考，作业 `3979539` 的已完成结果提供本次直接比较依据。

记录改进模型的结构、可训练参数量、向量维度、实际难负候选及计算量，基线对应信息引用已有记录。成绩变化的归因范围为完整组合结构，结论范围为 seed=42 的本次结果。

## 10 评价与 checkpoint 选择

### 10.1 候选库与指标

测试查询为 `val_60.csv` 中的 1,649 张街景，卫星候选为同一清单中的 1,649 张卫星图像。图片 ID 决定唯一正确候选，全部测试查询进入评价。

评价直接调用已完成 geo 复现使用的 `sample4geo.trainer.predict` 和 `sample4geo.evaluate.cvusa_and_cvact.calculate_scores`。`predict` 沿用原有 AMP 前向、L2 归一化、转换为 FP32 并在 GPU 保存特征的顺序；`calculate_scores` 沿用原有分块矩阵乘法、严格比较排名和 Recall 统计。基线向原预测函数提供原有全局 head 的输出；改进模型提供按可学习比例缩放并拼接后的最终向量，由原预测函数完成评价归一化。

原 `predict` 增加可选的特征提取回调，以接收改进模型的视角和样本 ID，并收集诊断量。原 `calculate_scores` 增加可选的详细结果返回接口，在同一次计算中返回四项 Recall、逐查询排名和完整相似度矩阵。两个函数的默认调用方式和原有计算规则保持不变。训练、难负候选生成、独立评价及已有基线评价共用这些原有函数。

评价过程中每个模型只使用所选 checkpoint 的最终向量和参数。候选集合保持完整，测试街景的真实经纬度及损伤标签只用于结果关联或分析，不参与候选筛选或分数计算。融合比例仅由训练集优化，每轮测试不单独调整该比例。

每个完整训练轮次结束后，在测试集上计算最终检索向量的：

- 主指标：R@1。
- 辅助指标：R@5、R@10、R@1%。

沿用作者排名定义：某个查询的排名值为“相似度严格大于正确卫星相似度的候选数量”；排名值小于 K 时记为 R@K 命中。R@1% 的 K 使用 `floor(1649/100)=16`。

对于 `Q=1649` 个查询，统一报告百分比：

\[
rank_i=\sum_j\mathbf{1}[s_{ij}>s_{i,gt(i)}],\qquad
R@K=\frac{100}{Q}\sum_i\mathbf{1}[rank_i<K].
\]

输出 Top-K 清单时，分数相同的候选按固定卫星 ID 顺序排列，同时记录相同分数数量。上述清单排序与作者指标的相同分数处理分别保存定义。

### 10.2 最佳权重

改进模型每个完整训练轮次结束后评价一次。沿用原复现的 `best_score=0` 初始值，按最终组合向量的测试集 R@1 严格提高时保存 best，出现相同 R@1 时保留较早的 checkpoint。全部轮次均未产生 R@1 大于 0 的权重时，沿用原复现的报错行为。基线使用其已保存的 best checkpoint。

训练结束后重新加载该轮 best，在完整测试集上重新计算四项指标。最终 R@5、R@10 和 R@1% 均来自该 checkpoint。

保存 best 和训练结束两份模型权重，包含融合参数、共用 `logit_scale`、模型结构配置、seed、已完成轮次、优化步数及划分标识，支持完整评价和参数分析。训练日志记录学习率与 GradScaler scale；两份权重文件用于评价，文件名分别为 `best.pth`、`last.pth`。

### 10.3 与已完成 geo 结果比较

分别列出作业 `3979539` 的基线成绩与本次改进模型 best checkpoint 的 R@1、R@5、R@10、R@1%，计算每项指标的百分点差。报告本次 seed=42 的单次训练结果及对应融合比例。

使用基线已有的逐查询预测记录，在同一测试 ID 清单上统计以下数量；记录不完整时，使用已保存的基线 best checkpoint 补充评估：

- baseline 错误、组合模型正确的数量。
- baseline 正确、组合模型错误的数量。
- 两种模型同时正确、同时错误的数量。

组合模型的全局分支和空间分支在最终 best checkpoint 上输出诊断指标，不额外选择各分支自己的最佳权重。结论依据完整模型的最终向量与对应统计。

记录预测耗时、峰值 GPU 显存及特征维度。网络容量、向量维度、可学习融合和采样结果的变化均属于整体结构变化，比较结果不单独归因于 CVFT 或残差模块。

## 11 实现和运行验收

以下项目用于核验模型计算、参数更新和评价是否符合设计，涉及 GPU 的检查使用真实数据在计算节点执行。独立验收作业不列入本次运行安排；现有验收脚本的具体行为见第 14.4 节。本次正式实验按第 9.4 节执行。

1. 使用真实 CVIAN 图片核对预处理尺寸、空间特征形状、4×16 和 8×8 的展平及恢复顺序。
2. 使用真实训练 batch 检查 batch 内 ID 唯一性、同 ID 配对、两张 GPU 特征汇集及 64×64 InfoNCE logits。
3. 检查矩阵生成只依赖街景，卫星候选向量能够逐幅独立计算。
4. 检查 Sinkhorn 的 dtype、非负性、有限性、行列和，以及 `P^T G` 的索引方向。
5. 使用实际模型输出检查均值保持关系，验证搬移与展开保留的空间顺序。
6. 检查全局向量、空间向量和组合向量的范数，零范数和非有限值直接报错。
7. 使用实际前向输出验证平方根缩放、组合向量单位范数，以及组合向量点积等于 `alpha` 加权的两分支相似度之和。覆盖初始化与实际参数更新后的融合比例。
8. 执行真实 batch 的前向、反向和参数更新，确认共享 backbone、通道投影、矩阵生成、修正模块、`fusion_logit` 及共用 `logit_scale` 均获得有限且有效的梯度；同时核对唯一训练损失确实读取最终融合向量。
9. 在两张实际 GPU 上验证正式输入和完整 batch 的显存与主机内存，记录实际峰值。资源不足时停止并修订资源或配置，不静默改变 batch。
10. 保存并加载真实模型权重，核对融合参数、同一输入的预测和完整测试 ID 关联。
11. 训练、推理和较大规模特征计算通过 Slurm 在计算节点进行。正式运行使用离线数据与已有权重。
12. 核验各评价入口直接调用原复现的 `predict`、`calculate_scores`，难负候选沿用原 `calculate_nearest` 和 GPU 分数计算。检查完整查询与候选 ID 的对应关系、四项 Recall 及逐查询结果，保留权重保存加载检查。

## 12 文件组织与实验记录

模型、Dataset、训练入口、评估入口和配置均安排在 `ME CVDisaster/` 内。基线来源保存在 `Author CVDisaster/`，组合模型实现独立管理。

改进模型本次训练输出使用服务器仓库外的独立目录，按模型名称、seed=42 和 JobID 标识。基线继续引用作业 `3979539` 的已有输出。记录至少包含：

- 训练代码提交号、方案编号、完整启动参数与 Slurm 作业标识。
- 环境、硬件、输入权重 SHA256、划分文件 SHA256。
- 实际图像处理参数、训练 seed、采样配置和每轮样本 ID。
- 最终融合向量的 InfoNCE 损失、共用 `logit_scale`、学习率与优化步数。
- `fusion_logit`、`alpha`、其梯度与 checkpoint 中保存的最终融合比例。
- Sinkhorn 误差、矩阵行熵和最大元素，以及梯度幅度、缺失和非有限值记录。
- best 和训练结束 checkpoint，保存与加载核验结果。
- 训练前和每轮的训练集、测试集指标，best 对应的完整逐查询预测与正确候选排名。
- 训练时间、预测时间、GPU 显存和主机内存。
- 改进模型与已完成 geo 结果的逐项比较及异常说明。

JobID、实际输出路径、命令及成绩在实施和运行后写入实验记录。当前方案不预填未来运行结果。

## 13 方法依据

- CVFT：学习图像相关的搬移矩阵、ReLU、代价生成参数初始化、Sinkhorn 计算及空间特征展开。参考 [Optimal Feature Transport for Cross-View Image Geo-Localization](https://arxiv.org/html/1907.05021) 和 [作者核心实现](https://raw.githubusercontent.com/YujiaoShi/cross_view_localization_CVFT/ddf6c2f6d35c0103b9315d686a3135a1a09db5fd/script/ot_net.py)。
- DenseFlow：投影或搬移后，通过残差模块进一步处理空间特征，使用 InstanceNorm、ReLU 和 Kaiming 初始化。参考 [Learning Dense Flow Field for Highly-accurate Cross-view Camera Localization](https://arxiv.org/abs/2309.15556)、[ResidualBlock 与 RefineNet 源码](https://raw.githubusercontent.com/XianghuiZe/DenseFlow_localization/16d9f106e9cea4c04458e4cfb9be35f7fba5ece3/gen_BEV/VGG.py) 和 [RAFT 中的 InstanceNorm 配置](https://raw.githubusercontent.com/XianghuiZe/DenseFlow_localization/16d9f106e9cea4c04458e4cfb9be35f7fba5ece3/op_flow/raft.py)。本方案的一个 64 通道残差单元用于当前模型的 8×8 网格。
- 现有 geo：共享 ConvNeXt、CVUSA 预训练、图像预处理、InfoNCE、难负采样及评价规则。对应仓库中的 `Author CVDisaster/geolocalization/` 与 `slurm/train_geoloc.sbatch`。

4×16 街景源网格、8×8 卫星目标网格、5,120 维组合向量、可学习融合和空间修正模块的上述具体组合，属于本实验的网络设计。训练采用最终融合向量的作者双向 InfoNCE，CVIAN 上的收益通过本方案规定的训练和比较验证。

参考代码版本：CVFT 为 `YujiaoShi/cross_view_localization_CVFT` 的 `ddf6c2f6d35c0103b9315d686a3135a1a09db5fd`，核心文件为 `script/ot_net.py`；DenseFlow 为 `XianghuiZe/DenseFlow_localization` 的 `16d9f106e9cea4c04458e4cfb9be35f7fba5ece3`，相关文件为 `models_vigor.py`、`gen_BEV/VGG.py` 和 `op_flow/raft.py`。

## 14 代码接口与运行说明

### 14.1 模型和数据接口

`ME CVDisaster/geolocalization/model.py` 提供 `GeoModel`。训练调用 `model(ground, satellite, sample_ids=ids)`，返回两个字典。单视角调用 `model(image, view="ground", sample_ids=ids)` 或 `model(image, view="satellite", sample_ids=ids)`。

`ME CVDisaster/geolocalization/spatial.py` 提供 `FeatureTransport`、`SpatialRefinement`、CVFT 卷积初始化函数及 `create_spatial_modules`。各模块按第 5、6 节初始化，`create_spatial_modules` 负责初始化随机状态的保存和恢复，`GeoModel` 直接使用返回的模块。

返回字典包含 `global_features` 和 `descriptors`；组合模型还包含 `spatial_features` 和 `fusion_weight`，街景另外提供 `transport` 和 `transport_diagnostics`。`fusion_weight` 对每个样本记录同一个模型比例，供 DataParallel 汇集；`fusion_state()` 返回模型共用的标量记录。归一化、融合及损失公式见第 7、8 节。

空间向量按位置、通道顺序展开，搬移使用 `P.transpose(1,2) @ G`。`ME CVDisaster/geolocalization/data.py` 直接调用原复现的 Dataset、transforms 和默认 DataLoader worker 初始化；读取已有 CSV 并记录行数及 SHA256。共享数据和已有 CSV 仅供读取，具体预处理见第 3 节。

配置文件为 `ME CVDisaster/configs/geo_cvft_refine.json`。两张 GPU 使用 DataParallel 汇集特征后，在主卡计算全局 64×64 InfoNCE。正式训练入口和作业脚本仅接受 `geo_cvft_refine`、seed=42；基线模型接口用于读取已有权重进行评价。

配置中的 `baseline_reference` 记录作业 `3979539` 的训练提交、seed、成绩和 best checkpoint。已确认的权重路径为 `/scratch/liuzeyu/heyinxi/runs/geoloc-3979539/Author CVDisaster/geolocalization/cvdisaster/convnext_base.fb_in22k_ft_in1k_384/111318/weights_e10_79.8059.pth`。

### 14.2 输出文件

训练输出目录包含：

- `run.json`：完整配置、代码提交号、运行环境、GPU、初始权重与划分 SHA256、图像处理参数、实际调度长度，以及训练前训练集与测试集的四项 Recall 和 Sinkhorn 诊断。
- `steps.jsonl`：每步损失、学习率、logit scale、GradScaler scale、融合比例和梯度统计、Sinkhorn 误差、行熵及最大元素。
- `history.json`：每轮损失、训练集与测试集四项 Recall、融合比例、Sinkhorn 汇总、实际样本顺序、耗时和内存。
- `best.pth`、`last.pth`：按第 10.2 节保存的两份模型权重与评价所需元数据。
- `best_queries.csv`：每个查询一行，包含正确候选排名与分数、Top-1 结果、图片路径、Top-K ID 和分数列表、两个分支对正确候选及 Top-1 候选的分数、逐查询 Sinkhorn 诊断。
- `result.json`：best 重新加载后的四项 Recall、两个分支的诊断成绩、最终融合比例、耗时和内存、保存加载检查结果，以及与基线的指标差和正确性数量比较。
- `baseline_reference/result.json`、`baseline_reference/best_queries.csv`：已有基线权重的评价结果及逐查询结果，用于分析改进模型修正或新增的错误。

Top-K ID 和分数列表在 CSV 单元格内使用 JSON 格式；同分候选采用数值 ID 升序排列。评价特征和完整相似度矩阵仅在当前计算期间使用，分析结果按上述文件保存。

### 14.3 运行命令

训练、评估和验收入口要求 Slurm 计算节点环境。运行脚本为 `ME CVDisaster/scripts/experiments/geo_cvft_refine.sbatch`，每次只执行 `CVDISASTER_MODE` 指定的一项操作。

以下命令对应用户已确认的单次训练参数。实际 JobID 和使用的代码提交号写入本次提交记录。提交前确认账号、分区、存储空间和环境可用性。作业脚本的时间上限为 24 小时。

```bash
cd /scratch/liuzeyu/heyinxi/CVDisaster
export CVDISASTER_COMMIT=$(git rev-parse HEAD)

# 改进模型正式训练一次，seed=42。
bash "ME CVDisaster/scripts/experiments/submit_geo_cvft.sh"
```

同一个训练任务执行以下计算：训练前评价初始模型的测试集和训练集，生成难负候选并 shuffle；训练 10 个 epoch，每轮结束后评价测试集和训练集、生成难负候选、选择 best 并 shuffle；训练结束后重新加载 best 计算完整指标并检查保存加载结果。随后读取作业 `3979539` 的已有 best checkpoint，调用同一套原复现评价函数，保存逐查询结果并计算比较统计。已有基线在该过程中的参数更新次数为零。历史成绩和本次成绩按实际数值记录并比较。

需要独立读取本次改进模型权重进行评估时，使用以下命令，将路径替换为实际输出目录：

```bash
export CVDISASTER_CHECKPOINT=/scratch/liuzeyu/heyinxi/runs/geo_cvft_refine-seed42-<JobID>/output/best.pth
export CVDISASTER_MODE=evaluate
unset SBATCH_GET_USER_ENV SLURM_GET_USER_ENV SBATCH_EXPORT SLURM_EXPORT_ENV
sbatch --export=ALL "ME CVDisaster/scripts/experiments/geo_cvft_refine.sbatch"
```

本次结果比较按照第 10.3 节，直接使用作业 `3979539` 的已有结果和本次改进模型结果，比较统计统一写入 `result.json`。需要单独导出已有逐查询记录的比较报告时，调用 `compare.py --run <改进模型output目录> --baseline-queries <基线best_queries.csv> --output <仓库外的新目录>`；该工具只读取结果文件。

### 14.4 验收脚本与作业的用途

验收脚本检查代码是否正确完成指定计算，检查结果用于发现尺寸、数值、梯度、内存及保存加载问题。正式训练结果用于评价模型的检索效果，两者分别记录。

`ME CVDisaster/validate.py` 使用真实 CVIAN 图片和已有 CVUSA 初始权重，检查训练 batch 64、评价 batch 128、两张 A100、独立单视角预测、矩阵和特征顺序、均值关系、向量范数、融合公式、最终向量损失及各模块梯度。它会针对配置中列出的每种模型完成一次参数更新，随后执行完整测试集评估、保存和加载检查，输出验收 JSON 与验收权重。当前默认配置包含 baseline 和改进模型，因此调用该入口会检查这两种模型，并分别更新一次参数。

`CVDISASTER_MODE=validate` 会通过同一 Slurm 脚本申请计算节点并执行 `validate.py`。这是独立的计算作业，需要 GPU 资源，其输出权重和成绩只用于检查计算。本次运行安排未包含该独立作业，验收权重不作为本次正式训练的初始权重。

评价函数由 `Author CVDisaster/geolocalization/sample4geo/trainer.py` 的 `predict` 和 `sample4geo/evaluate/cvusa_and_cvact.py` 的 `calculate_scores` 提供。`ME CVDisaster/geolocalization/evaluation.py` 负责模型视角适配、诊断量收集和结果文件输出。权重保存加载检查继续比较保存时与重新加载后的融合参数、查询排名及 Recall。

第 11 节列出的检查项目描述需要核验的计算内容；实际执行过的检查、作业编号和结果分别记录。正式作业的运行结果以对应日志和输出文件为准。
