# ME CVDisaster

这是本项目的改进实验目录，与 `Author CVDisaster/` 平行。

`geo_cvft_refine` 使用共享 ConvNeXt、CVFT 特征搬移和街景残差修正，通过可学习比例组合成 5,120 维向量，使用最终向量的双向 InfoNCE 训练。`baseline` 使用相同数据处理与训练参数，输出 1,024 维全局向量。两个模型直接复用作者目录中的图像增强、Dataset 和难负采样实现。

实验结构与参数见 [实验方案](docs/Geo_CVFT_空间特征修正实验方案.md)，代码接口、运行命令及验收要求见 [运行说明](docs/Geo_CVFT_代码运行说明.md)。

## 目录约定

- `scripts/reproduction/`：复现作者实验的脚本和检查脚本。
- `scripts/experiments/`：后续改进实验的运行脚本。
- `configs/`：实验配置和参数记录。
- `docs/`：本目录专属的实验方案和实验记录。
- `geolocalization/`：模型、损失、数据接口、训练与评价代码。
- `train.py`：指定模型与 seed 的训练入口。
- `evaluate.py`：已有实验 checkpoint 的独立评价入口。
- `validate.py`：真实图片、完整 batch、两张 A100 的验收入口。
- `compare.py`：六次正式训练结果的成对比较入口。
- `train_results/log/`、`train_results/raw/`：训练日志与训练产物，仅保存在服务器。
- `test_results/log/`、`test_results/raw/`：测试日志与测试产物，仅保存在服务器。

所有大文件、数据集、权重和运行结果均不得进入 GitHub。每个实验目录必须记录代码提交号、环境、数据划分、运行命令和结果路径。
