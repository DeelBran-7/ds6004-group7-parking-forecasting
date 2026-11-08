# RQ2 余位门槛扩展：结果与复现

本目录保存“达标后选最近”的新成对策略，与原标准两年实验比较。它是已查看旧测试结果后的回顾性扩展，原 C−C0 结果保留。

**第一版没有通过验证段经验约束。** 所有预定 k 都不满足要求；k=3 仅是预先固定的诊断兜底，不作为合格改进策略。预测版没有优于同门槛的当前版。后续带误差余量的连续探索见 [第二阶段](../rq2_buffered_gate/README.md)，不是独立确认。

先读 [固定实验方案](../../docs/rq2_gate_protocol.md) 和 [图文报告](../../reports/rq2_gate/README.md)。

## 结果文件

| 文件 | 用途 |
| --- | --- |
| [validation_metrics.csv](validation_metrics.csv) | 同一验证候选池中的原 C0 与门槛策略结果 |
| [threshold_selection.csv](threshold_selection.csv) | 固定门槛网格的经验约束、距离及选择结果 |
| [recommendation_metrics.csv](recommendation_metrics.csv) | 锁定门槛下的当前版、预测版与旧策略；两时长、两种候选口径 |
| [paired_policy_comparisons.csv](paired_policy_comparisons.csv) | 预测增量、规则增量、总变化三类配对比较 |
| [threshold_sensitivity.csv](threshold_sensitivity.csv) | 全部门槛的测试敏感性，不用于重选主门槛 |
| [daily_policy_diagnostics.csv](daily_policy_diagnostics.csv) | 逐日首选变化、增加/避免满位与距离差 |
| [audit.json](audit.json) | 输入指纹、方法、匹配检查和结果解读限制 |
| [gate_outcomes.png](gate_outcomes.png) | 距离与到达结局图 |
| [gate_paired_effects.png](gate_paired_effects.png) | 配对效应与不确定性 |

预测版的余位为 `(1−M1占用率) × 出发时容量`，是点估计，不是概率或置信下界。达标门槛不保证到达有位。满位（=0）、低余位（<3）与比例近满（<容量10%）分别报告。

`FORECAST_GATE−CURRENT_GATE` 才衡量同规则下预测的额外价值；`CURRENT_GATE−C0` 衡量规则改变；`FORECAST_GATE−C0` 是总变化，不能全部归因于预测。未显著变差不构成非劣性证明，全零差异不构成等效证明。

## 复现

使用 Python 3.11，安装 requirements-evaluation.txt。需要本地已有 `outputs/history_comparison/two_year_standard/` 的完整预测和推荐产物；若缺失，先按 [标准两年运行命令](../history_comparison/README.md) 生成。

```sh
python scripts/run_rq2_gate_experiment.py \
  --study outputs/history_comparison/two_year_standard \
  --output outputs/rq2_gate
```

本轮只重算推荐，不训练或校准新模型。固定 k 网格为 1、3、5、10、20；只用验证段 CURRENT_GATE 的观察结果与原 C0 比较，两个时长的满位及低余位率均不高于 C0 的门槛才进入可行集合，再按加权平均距离选择。预测版与测试结果都不参与选择。完整排名保存在本地 `outputs/rq2_gate/selected_rankings.csv`，未上传。

配对检验按出发日期分块，10,000 次 bootstrap 与双侧符号翻转；整张配对表做 Holm 校正，95% 区间不做多重校正。区间和 p 值依赖独立、可交换日期块等假设。公开目录仅放小型汇总、审计和两张图。
