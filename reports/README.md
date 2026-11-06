# 测评报告

这里集中放置可以直接阅读、展示和分享的报告。

| 报告 | 内容 | 格式 |
| --- | --- | --- |
| 最新：一年与两年对比 | 修正版数据的公平历史长度比较，以及完整两年 60/20/20 测评 | [图文报告](history_comparison/README.md) · [6 页 PDF](history_comparison/history_comparison.pdf) |
| 年度 A → B → C 测评 | 8 个停车场、365 天数据；预测、到达风险、推荐效果及数据质量 | [图文报告](annual/README.md) · [PDF](annual/annual_evaluation.pdf) |

最新报告数字对应 [`examples/history_comparison/`](../examples/history_comparison/README.md)，旧年度报告对应 [`examples/annual/`](../examples/annual/README.md)。逐时点预测与完整回放明细需要在本机重跑，位于 `outputs/`。旧报告基于旧清洗版本，保留其原始结论。

## 重新生成图与 PDF

以下命令只使用已有汇总，不需要原始年表，也不重训模型：

```sh
python -m pip install -r requirements-report.txt
python scripts/build_annual_report.py --report-dir reports/history_comparison \
  --evaluation-dir examples/history_comparison/two_year_standard/evaluation \
  --pdf-name history_comparison.pdf
```

最新报告的两张历史长度图随仓库保存于 `reports/history_comparison/figures/` 与 `examples/history_comparison/comparison/`。从完整本地预测重算比较及图表的命令见 [结果索引](../examples/history_comparison/README.md)。报告文字与表格由人工核对后保存为 Markdown，PDF 导出不会自动刷新文字中的数字。

重新生成旧年度报告：

```sh
python scripts/build_report_figures.py
python scripts/build_annual_report.py
```

PDF 内容来自 `reports/annual/README.md`；更新数据后，先核对并更新报告文字与表格，再导出 PDF。PDF 导出需要可嵌入的中文 TrueType 字体，macOS 默认使用系统 Arial Unicode；其他系统可通过 `--font /path/to/font.ttf` 指定。所有图表都保存在 `reports/annual/figures/`，可直接用于组内讨论。

[返回项目首页](../README.md)
