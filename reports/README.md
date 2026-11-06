# 测评报告

这里集中放置可以直接阅读、展示和分享的报告。

| 报告 | 内容 | 格式 |
| --- | --- | --- |
| 年度 A → B → C 测评 | 8 个停车场、365 天数据；预测、到达风险、推荐效果及数据质量 | [图文报告](annual/README.md) · [PDF](annual/annual_evaluation.pdf) |

报告数字对应 [`examples/annual/`](../examples/annual/README.md) 的汇总表。逐时点预测与完整回放明细需要在本机重跑，位于 `outputs/annual/`。

## 重新生成图与 PDF

以下命令只使用已有汇总，不需要原始年表，也不重训模型：

```sh
python -m pip install -r requirements-report.txt
python scripts/build_report_figures.py
python scripts/build_annual_report.py
```

PDF 内容来自 `reports/annual/README.md`；更新数据后，先核对并更新报告文字与表格，再导出 PDF。PDF 导出需要可嵌入的中文 TrueType 字体，macOS 默认使用系统 Arial Unicode；其他系统可通过 `--font /path/to/font.ttf` 指定。所有图表都保存在 `reports/annual/figures/`，可直接用于组内讨论。

[返回项目首页](../README.md)
