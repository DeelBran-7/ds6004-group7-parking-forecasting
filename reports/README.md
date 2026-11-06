# 测评报告

这里集中放置可以直接阅读、展示和分享的报告。

| 报告 | 内容 | 格式 |
| --- | --- | --- |
| 最新：RQ 补充测评 | 完整后一年假日测试；原标准测试晚高峰、周末；保留有利与不利发现 | [图文报告](rq_supplement/README.md) · [4 页 PDF](rq_supplement/rq_supplement.pdf) |
| 一年与两年对比 | 修正版数据的公平历史长度比较，以及完整两年 60/20/20 测评 | [图文报告](history_comparison/README.md) · [6 页 PDF](history_comparison/history_comparison.pdf) |
| 年度 A → B → C 测评 | 8 个停车场、365 天数据；预测、到达风险、推荐效果及数据质量 | [图文报告](annual/README.md) · [PDF](annual/annual_evaluation.pdf) |

RQ 补充对应 [`examples/rq_supplement/`](../examples/rq_supplement/README.md)，历史长度与标准两年对应 [`examples/history_comparison/`](../examples/history_comparison/README.md)，旧年度对应 [`examples/annual/`](../examples/annual/README.md)。逐时点预测与完整回放明细需要在本机重跑，位于 `outputs/`。旧报告保留原始实验、清洗版本与结论。

## 重新生成图与 PDF

以下命令只使用已有汇总，不需要原始年表，也不重训模型：

```sh
python -m pip install -r requirements-report.txt
python scripts/build_annual_report.py --report-dir reports/rq_supplement \
  --skip-evaluation-figures --pdf-name rq_supplement.pdf
python scripts/build_annual_report.py --report-dir reports/history_comparison \
  --evaluation-dir examples/history_comparison/two_year_standard/evaluation \
  --pdf-name history_comparison.pdf
```

RQ 补充的 4 张图位于 `reports/rq_supplement/figures/`，历史长度图位于 `reports/history_comparison/figures/`。从完整本地预测重算的命令见 [RQ 复现](../examples/rq_supplement/README.md) 与 [历史长度复现](../examples/history_comparison/README.md)。报告文字与表格由人工核对后保存为 Markdown，PDF 导出不会自动刷新文字中的数字。

重新生成旧年度报告：

```sh
python scripts/build_report_figures.py
python scripts/build_annual_report.py
```

PDF 内容来自 `reports/annual/README.md`；更新数据后，先核对并更新报告文字与表格，再导出 PDF。PDF 导出需要可嵌入的中文 TrueType 字体，macOS 默认使用系统 Arial Unicode；其他系统可通过 `--font /path/to/font.ttf` 指定。所有图表都保存在 `reports/annual/figures/`，可直接用于组内讨论。

[返回项目首页](../README.md)
