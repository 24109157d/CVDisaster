# 文档目录

[实验方案](实验方案.md) 是当前实验的主要依据，集中说明原 geo 设置、局部结构、训练损失、固定 40% 融合、评价规则和参数。

| 位置 | 内容与用途 |
|---|---|
| [实验方案.md](实验方案.md) | 当前采用的完整方案 |
| [records/](records/) | 实际作业的源码版本、资源、参数和完成成绩；用于核对一次实验实际运行了什么 |
| [diagnostics/](diagnostics/) | CPU 小样本与无更新诊断的结果、样本范围和结论限制 |
| [archive/](archive/) | CVFT 历史方案与作业记录；用于理解已有实验及读取旧模型 |
| [templates/](templates/) | 新增实验记录时使用的模板 |

已完成的候选匹配实验为 [作业 4309752](records/Geo_partial_match_gate_2026-11-14_提交记录.md)。固定 40% 的训练配置以当前主方案和 `configs/geo_partial_match_fixed40.json` 为准，实际任务提交后另行保存对应记录。
