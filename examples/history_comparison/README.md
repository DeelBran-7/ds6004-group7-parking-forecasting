# 一年与两年历史数据的对比结果

本目录包含基于最新 `two_year_training_corrected_proposal.csv` 的三组 A → B → C 汇总，以及共同样本对比与数据质量审计。全部组使用同一份修正版数据，一年组截取其最近一年。

## 实验目录

| 目录 | 历史范围 | 验证开始 | 测试开始 | 用途 |
| --- | --- | --- | --- | --- |
| [one_year](one_year/) | 2025-11-05 至 2026-11-04 | 2026-06-12 | 2026-08-24 | 最近一年对照组 |
| [two_year_shared](two_year_shared/) | 2024-11-05 至 2026-11-04 | 2026-06-12 | 2026-08-24 | 固定验证、测试日期，增加较早训练历史 |
| [two_year_standard](two_year_standard/) | 2024-11-05 至 2026-11-04 | 2026-01-17 | 2026-06-12 | 完整两年按 60/20/20 切分的独立实验 |

判断历史长度的效果应阅读 `one_year` 与 `two_year_shared` 的共同样本比较。两组的验证、测试各 73 天，训练日期分别为 219 / 584 天，两个边界前各隔离 30 分钟。这里比较的是可用历史窗口长度，训练段不包含验证和测试日期。

`two_year_standard` 的验证、测试各 146 天，测试含 2 个公共假日日期。共同测试段含 0 个公共假日日期。不同留出时间段的结果应分别解释。

## 查看结果

| 位置 | 内容 |
| --- | --- |
| [comparison/](comparison/) | 两年共同边界组减去一年组的配对误差、风险及推荐差值，含两张比较图 |
| [data_quality/README.md](data_quality/README.md) | 新数据有效性、标签核对、旧年度文件差异、时间切分与来源可追溯性 |
| [experiment_manifest.json](experiment_manifest.json) | 原始文件哈希、数据范围、共同边界与实验设定 |
| [run_environment.json](run_environment.json) | 本次运行的软件版本与导出环境 |

每组的 `forecast/` 保存输入审计和单模型诊断，`risk/` 保存到达风险、校准及转换汇总，`recommendation/` 保存验证权重和回放汇总，`evaluation/` 保存共同样本评分、统计比较与三张效果图。

MAE 在 CSV 中使用 0–1 占用率比例，展示为百分点时乘 100。Brier 是概率平方误差。真正满位为空位 = 0，低余位为空位 < 3，近满为空位 < 容量 10%。推荐距离为直线距离。`comparison/` 的差值方向为 `two_year_shared - one_year`，负误差差值表示两年组较好。

原始约 190 MB 年表由组内提供；逐时点预测、模型文件、推荐排名和逐案审计保存在本机 `outputs/history_comparison/`，本目录仅公开汇总和图表。复现时由组员自行提供原始数据。

## 复现命令

在仓库根目录安装 `requirements-evaluation.txt` 后，完整运行三组及对比：

```sh
python scripts/run_history_comparison.py \
  --history /path/to/two_year_training_corrected_proposal.csv \
  --output outputs/history_comparison
```

已有三组逐时点输出时，只重算一年与两年的配对统计和图表：

```sh
python scripts/compare_history_results.py \
  --one-year outputs/history_comparison/one_year \
  --two-year outputs/history_comparison/two_year_shared \
  --output outputs/history_comparison/comparison
```

单独重跑完整两年的 proposal 标准 60/20/20 实验：

```sh
python run_study.py --history /path/to/two_year_training_corrected_proposal.csv \
  --holidays data/sg_public_holidays_2020_2027.csv \
  --metadata data/chinatown_annual_carparks.csv \
  --destination data/chinatown_destination.json \
  --output outputs/history_comparison/two_year_standard
```

共享边界由最近一年时间网格的 60% / 80% 位置确定。两年共享组因扩展训练历史而约为 80/10/10，专用于历史长度比较；标准两年组另按全历史 60/20/20 划分。两者用途不同，不横向排名测试分数。

完整脚本会重建模型、风险校准、推荐、评估和历史长度比较；`data_quality/` 的专项新旧数据核查是本次独立审计的记录，不由此命令自动重建。报告 PDF 从已核对的 Markdown 导出，见 [报告构建说明](../../reports/README.md)。

[返回项目首页](../../README.md)
