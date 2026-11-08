# RQ2 扩展第二阶段：经验误差余量

**推荐规则让距离缩短约 60 米，但同规则下预测的额外收益不足 0.1 米。** 满位事件未因加入预测减少，不能据此宣称原 RQ2 明显提升。与旧 C0 相比，近满事件增加，必须与距离收益一起报告。

这是查看旧结果及[第一阶段](../rq2_gate/README.md)后进行的一次回顾性扩展，不是独立确认。原 C−C0 主实验不变。阅读[图文报告](../../reports/rq2_gate/README.md)、[固定方案](../../docs/rq2_gate_protocol.md)。

## 锁定的方法与选择

早验证段（2026-01-17 至 03-31）提供 108,338 条共同校准样本；按停车场、预测时长及信号分别计算空位高估残差的经验分位数，将非负余量从原估计空位中扣除。当前版和预测版都采用这一方法。当前版不使用 M1，但会利用历史变化残差，不能称作原 C0。

晚验证段（2026-04-01 至 06-11）仅用 CURRENT_BUFFERED_GATE 选择共享参数：两个时长的观察满位及少于 3 位事件率均不高于同池 C0，再最小化合并平均距离。固定网格 q={0.90,0.95,0.975,0.99}、k={3,5,10,20}，16 组全部可行，锁定 q=0.90、k=3。**该段 C0 和所有当前余量组合的满位及低余位均为零，筛选证据较弱，不能视为安全认证。**

满足“扣余量后估计空位 ≥ k”时按距离排序；否则按扣余量后的估计空位降序兜底。使用出发时容量，候选资格仍按真实当前空位判断。经验分位数不是风险概率、预测区间或覆盖保证。早晚验证分开、边界隔离 30 分钟，测试从 2026-06-12 至 11-04。

## 主要结果

以下为出发有位口径。15 / 30 分钟分别有 13,597 / 13,586 个匹配场景，覆盖 146 个出发日期。

| 策略（15 / 30 分钟） | 平均直线距离（米） | 满位次数 | 少于3位次数 | 近满次数 |
| --- | --- | --- | --- | --- |
| 原 C0 | 262.075 / 262.096 | 1 / 2 | 1 / 2 | 2 / 2 |
| 当前信号 + 余量 | 202.212 / 202.281 | 0 / 1 | 1 / 2 | 90 / 77 |
| 预测信号 + 余量 | 202.189 / 202.192 | 0 / 1 | 1 / 3 | 99 / 100 |

近满是到达空位少于到达容量的 10%，包含满位，但不等于实际停车失败；少于 3 位也是单独结局。两类余量策略相对 C0 的距离下降和近满增加均有 Holm p=0.0042；满位差异不显著，不能推断风险非劣。

预测版相对当前余量版：首选仅改变 10 / 36 次，距离少 0.023 / 0.089 米；后者 Holm p≈0.0125，实际幅度仍很小。满位次数完全相同，30 分钟低余位多 1 次。相应比较隔离的是“信号及其各自经验余量”的增量。

第一阶段用完整验证期选参；本阶段用后半验证期，不能把两次筛选结果不同完全归因于加入余量。当前测试集上的方法搜索到此停止；全部网格结果与失败诊断保留。

## 文件索引

| 文件 | 用途 |
| --- | --- |
| [calibration_margins.csv](calibration_margins.csv) | 所有 q、信号、停车场、时长的经验余量 |
| [calibration_audit.json](calibration_audit.json) | 共同样本、训练 / 校准 / 选择时间边界 |
| [late_validation_metrics.csv](late_validation_metrics.csv) | 晚验证结果 |
| [parameter_selection.csv](parameter_selection.csv) | 全部 16 组经验条件及选择记录 |
| [recommendation_metrics.csv](recommendation_metrics.csv) | 两时长、两口径、旧六策略与新两策略 |
| [paired_policy_comparisons.csv](paired_policy_comparisons.csv) | 预测增量、规则 / 历史增量、总变化的配对统计 |
| [parameter_sensitivity.csv](parameter_sensitivity.csv) | 全部固定组合的测试敏感性，不用于再选参数 |
| [daily_policy_diagnostics.csv](daily_policy_diagnostics.csv) | 逐日首选改变、增加 / 避免满位、距离变化 |
| [audit.json](audit.json) | 输入指纹、旧策略复现、选择方法及解释限制 |
| [buffered_gate_outcomes.png](buffered_gate_outcomes.png) | 全策略距离及三类到达结局 |
| [buffered_gate_paired_effects.png](buffered_gate_paired_effects.png) | 预测版相对当前余量版的效应与区间 |

日期块 bootstrap 10,000 次，双侧日期块符号翻转，整张配对表单独 Holm 校正；95% 区间不做多重校正。相邻日期依赖、稀少事件及回顾性连续探索限制推断，不能把全零差异当等效证据。第二阶段 p 值没有合并校正此前的全部探索。

## 复现

Python 3.11，安装 requirements-evaluation.txt；先按[历史实验命令](../history_comparison/README.md)生成完整标准两年预测与推荐结果，然后运行：

```sh
python scripts/run_rq2_buffered_gate.py \
  --study outputs/history_comparison/two_year_standard \
  --output outputs/rq2_buffered_gate
```

本轮不重训预测模型。完整场景排名留在本地 outputs/rq2_buffered_gate/selected_rankings.csv，未上传；公开目录为小型汇总、审计和图。
