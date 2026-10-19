# ME CVDisaster

这是本项目自己的复现与后续改进目录，与 `Author CVDisaster/` 平行。

当前阶段只建立目录和记录接口，不复制或修改作者实现。复现基线确认后，新增训练、评估、配置和改进代码放在本目录；作者代码只作为可追溯的基线来源。

## 目录约定

- `scripts/reproduction/`：复现作者实验的脚本和检查脚本。
- `scripts/experiments/`：后续改进实验的运行脚本。
- `configs/`：实验配置和参数记录。
- `docs/`：本目录专属的实验方案和实验记录。
- `train_results/log/`、`train_results/raw/`：训练日志与训练产物，仅保存在服务器。
- `test_results/log/`、`test_results/raw/`：测试日志与测试产物，仅保存在服务器。

所有大文件、数据集、权重和运行结果均不得进入 GitHub。每个实验目录必须记录代码提交号、环境、数据划分、运行命令和结果路径。
