# RQ 补充实验结果与复现

本目录保存 2026-11-06 的假日时间留出、晚高峰和周末补充分析。它们是在查看既有研究后固定方案再执行的回顾性分析，不替换原两年标准主实验，也不是全新独立确认。

先读 [固定方案](../../docs/rq_supplement_protocol.md) 与 [图文结果报告](../../reports/rq_supplement/README.md)。

| 目录 / 文件 | 内容 |
| --- | --- |
| [holiday/](holiday/) | 训练含假日与非假日，完整后一年测试；原 M0/M1 参数不变，只重训 4 个回归模型 |
| [holiday/forecast_metrics.csv](holiday/forecast_metrics.csv) | 四个模型、15/30 分钟、整体/假日/非假日共同样本 MAE/RMSE |
| [holiday/paired_holiday_comparisons.csv](holiday/paired_holiday_comparisons.csv) | 按目标日期整块的 12 项配对比较，整表 Holm 校正 |
| [holiday/holiday_episode_comparisons.csv](holiday/holiday_episode_comparisons.csv) | 连续假日合并成 9 组后的 4 项敏感性比较，单独整表 Holm 校正 |
| [holiday/holiday_daily_metrics.csv](holiday/holiday_daily_metrics.csv) | 12 个假日逐日的 M0/M1 误差、名称、分组和样本数 |
| [holiday/experiment_manifest.json](holiday/experiment_manifest.json) | 源文件/日历/预测代码哈希、切分和实际训练假日样本数 |
| [subgroups/](subgroups/) | 不重训，复用 two_year_standard 的预测与推荐结果；overall、17:00-20:00、周末三组 |
| [subgroups/paired_forecast_comparisons.csv](subgroups/paired_forecast_comparisons.csv) | 各子组内的 M0−B0 / M1−M0 比较，不是子组间交互检验 |
| [subgroups/paired_policy_comparisons.csv](subgroups/paired_policy_comparisons.csv) | C−C0 与直接风险策略−C0；真正满位仍为主要事件 |
| [run_environment.json](run_environment.json) | 软件环境 |

CSV 的 MAE 使用占用率比例，图与报告乘 100 显示百分点。`estimate` 为候选减参考，误差/风险/距离负差值表示下降；是否更好需结合多个结局解释。95% 区间不做多重校正，p 值按各整张 CSV 的全部有效检验做 Holm 校正，所以补充表中的 overall 校正 p 可能与旧主实验不同，原始效应估计不变。

## 运行假日时间留出

Python 3.11，安装仓库 requirements-evaluation.txt。组内提供原始两年 CSV，不包含在公开仓库中：

```sh
python scripts/run_holiday_holdout.py \
  --history /path/to/two_year_training_corrected_proposal.csv \
  --output outputs/rq_supplement/holiday
```

切分固定：训练 2024-11-05 至 2025-09-04，验证 2025-09-05 至 2025-11-04，测试 2025-11-05 至 2026-11-04；边界各隔离 30 分钟。训练含 11 个假日日期，验证 1 个，测试 12 个；连续假日组成 9 个测试分组。

在相同数据与预测代码下重算已有预测的统计和图表，加 `--reuse-predictions`。脚本会核对源数据、日历、预测代码哈希以及切分边界。

## 运行晚高峰 / 周末子组

需要本地已有标准两年逐时点预测和推荐排名；只有公开汇总 CSV 不能重建子组。缺失时，先按 [三组实验说明](../history_comparison/README.md) 生成 two_year_standard。

```sh
python scripts/analyze_rq_subgroups.py \
  --study-dir outputs/history_comparison/two_year_standard \
  --output outputs/rq_supplement/subgroups
```

晚高峰按出发时间 17:00 ≤ t < 20:00；周末为周六/周日，二者可重叠。脚本校验 overall 样本和指标复现既有发布结果。全部子组、两种候选口径、不利结果与零事件均保留。

## 图与报告

本目录保存 4 张 PNG；报告将它们集中展示。PDF 无须重训即可导出：

```sh
python -m pip install -r requirements-report.txt
python scripts/build_annual_report.py --report-dir reports/rq_supplement \
  --skip-evaluation-figures --pdf-name rq_supplement.pdf
```

报告文字来自已人工核对的 Markdown，导出不会自动更新数字。完整预测和模型保存在 outputs/rq_supplement/，Git 忽略。
