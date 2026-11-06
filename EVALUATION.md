# Part C：评估与完整流程

`run_study.py` 顺序运行 A 占用率预测与到达风险预测、B 推荐回放、C 评价。C 读取独立测试段，核对 A/B 的决策时点、到达标签、所用预测值和距离权重，再生成可直接用于报告的汇总和三张 PNG。

## C 负责什么

| 评价内容 | 输出 | 口径 |
| --- | --- | --- |
| 预测准确性 | `forecast_metrics.csv` | B0/B1/M0/M1 共同可评分样本的 MAE、RMSE、近满 precision/recall；另分近满、假日和非假日 |
| 停车场差异 | `forecast_metrics_by_carpark.csv` | 每场分别比较，同一场内仍要求模型共同样本 |
| 到达风险 | `risk_probability_metrics.csv` | 满位和少于 3 个空位分开；Brier、log loss、AP、AUROC；另报出发时有位子集 |
| 概率是否可信 | `risk_reliability.csv` | 独立测试段按概率十分位分箱，给出样本数、平均预测概率和真实事件率 |
| 推荐效果 | `recommendation_metrics.csv` | Top1 满位、近满、低余位；Top3 至少一场有位；直线距离与相对最近场的额外距离 |
| 配对比较 | `paired_*_comparisons.csv` | 相同样本的模型误差差值、风险分数差值，以及 C/风险策略相对 C0 的结果差值 |
| 实验审计 | `evaluation_audit.json` | 测试范围、样本量、重采样设定及假设 |

模型评分要求有效出发点和同一连续段内的有效到达标签。推荐回放根据有效的出发、到达端点评价全部策略；短暂的中间缺测可以导致两者分母不同。模型预测缺失时在共同支持集上比较，并记录覆盖率。若上周基准 B1 完全不可用（如三天 pilot），明确标为 unavailable，不用空值伪造结果。

## 对老师问题 2 的回答

预测的是 **出发时看到有位，到达时是否变成无位** 的概率。直接风险模型除了占用率，还使用容量、空位、历史变化和日历特征。满位定义为真实到达空位为 0；安全余量不足定义为空位少于 3；近满定义为空位严格少于容量的 10%。三者分别报告。

预测概率在按时间独立的验证段做校准，测试段仅用于最后评分。风险模型的容量特征使“26 位停车场”和“583 位停车场”的同样占用率不必对应相同风险。图中的概率是实测事件频率的估计，不能保证某名司机到达后一定有位。15 分钟快照也无法观察两个快照之间每一次进入、离开及占位竞争。

## 统计方法

以新加坡当地的**决策日期**为块，按日成对抽样 10,000 次，保留当天所有停车场、时点及相同策略的配对关系。报告差值的 95% percentile bootstrap 区间，并用按日成对符号翻转计算双侧 p 值。每张比较表内的有效 p 值做 Holm 校正；置信区间未做多重比较校正。负差值表示误差、风险或距离较小；Top3 可行率差值则越大越好。

按日抽样保留日内相关性，但假设日期块可交换，符号检验还要求独立且差值在零假设下对称。相邻日期的依赖可能影响这些近似；这里没有声称把十几万行当作十几万个独立样本。

若所有日期的配对差值均无变化，标记 `no_day_level_variation`，不输出可误读为“证明等效”的区间或 p 值。所有策略都零失败时，只能报告在本测试中未观察到首选满位，不能证明风险为零或策略等效。

## 年度实验的解释

以 [年度结果](ANNUAL_RUN.md) 为准。总体占用率误差降低，但近满样本的 15 分钟占用率 MAE 略差；直接风险模型相对占用率风险代理的 Brier 有改善，XGBoost 与 CatBoost 没有显著胜负。测试期间最近的 KAM 始终没有满位，推荐的失败率指标缺乏区分能力，不能将更好的预测分数直接解释为更好的推荐结果。

主分析固定保留 proposal 的目的地、时间切分和策略。以后若补充更多目的地、高需求日期、按时间滚动验证，必须作为新的验证设计单独报告；不能从这次测试结果倒过来挑选最有利的场景。

## 单独重跑 C

```sh
python -m pip install -r requirements-evaluation.txt
python evaluate.py --forecasts outputs/annual/forecast/predictions.csv \
  --risk outputs/annual/risk/risk_predictions.csv \
  --recommendation-dir outputs/annual/recommendation --output outputs/annual/evaluation
python -m unittest discover -s tests -v
```

输出图为 `forecast_mae.png`、`full_risk_calibration.png`、`recommendation_tradeoff.png`。
只需数值表时加 `--no-figures`；使用 `--seed` 和 `--repetitions` 调整统计重采样。
