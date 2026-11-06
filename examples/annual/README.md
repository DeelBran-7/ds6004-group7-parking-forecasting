# 年度实验汇总文件

本目录保存已完成年度实验的可核对结果。先阅读 [图文测评报告](../../reports/annual/README.md)，或下载 [PDF](../../reports/annual/annual_evaluation.pdf)。

| 目录／文件 | 包含内容 |
| --- | --- |
| [`forecast/`](forecast/) | A 占用率预测的原始单模型诊断与数据／切分审计 |
| [`risk/`](risk/) | A 到达风险评分、校准参数、可靠性分箱、状态转移与风险回放 |
| [`recommendation/`](recommendation/) | B 策略结果、固定目的地距离、验证选定权重及权重敏感性 |
| [`evaluation/`](evaluation/) | C 共同样本评分、分场评分、配对置信区间与检验，以及三张结果图 |
| [`data_quality/`](data_quality/README.md) | 年表有效率、按场和按时间段的风险分布，以及 3 条待核实容量变动 |
| [`run_environment.json`](run_environment.json) | 本次运行的 Python／依赖版本和测试记录 |

用于报告模型对比的误差以 C 的 [`forecast_metrics.csv`](evaluation/forecast_metrics.csv) 为准：它在模型共同可评分的样本上比较。A 的诊断表允许每个模型使用自身可用样本，分母可能不同。推荐回放按有效出发和到达端点评价，分母也不必与模型评分一致。

所有时间为新加坡时间；比例字段通常取值 0–1，转成百分比时乘 100。占用率 MAE 乘 100 后的单位是**百分点**。真正满位（空位为 0）、低余位（空位 <3）和近满（空位 <10% 容量）分别报告。

原始年度 CSV、完整预测明细及逐时点推荐排名未包含在本目录。按 [首页命令](../../README.md#跑完整年度实验) 可在持有原始年表的环境中重新生成。
