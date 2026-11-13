# 英文模型章节

面向最终小组 report 的模型部分，2026-11-13 整理。以数据组英文报告、此前 Poster、实际执行代码和已完成实验为依据；本次没有重新训练或更改实验结论。

**[下载可编辑 Word](DS6004_Group7_Model_Section_Report_EN.docx)** · [在线阅读全文](Model_Section_EN.md)

Word 经渲染检查为 8 页，包含 8 张表和 2 张图。内容涵盖输入与目标、特征工程、B0/B1/M0/M1、直接满位与低余位风险、Platt 校准、标准测试结果、假日与历史长度补充、推荐接口和局限。核心定位是模型贡献；推荐结果用于解释预测分数与实际决策效果的区别。

## 并入小组报告

建议接在 Data 部分之后。合并时统一章节、表格、图片和参考文献编号，并移除独立文档的标题信息与页眉页脚即可。第 6 节为补充实验，第 7 节后半为后续探索，不能当作原 Poster 主测试的同一结果；若篇幅有限，可将这些段落转至附录。

满位为 0 空位，模型低余位为 <3 空位，proposal 近满为 <10% 空位，三者已分开。数据组保留的 <5 / <10 字段不被当成该风险模型的目标。模型共同比较与推荐出发场景的分母也已分开。

## 文件与数值来源

| 文件 | 用途 |
| --- | --- |
| [Model_Section_EN.md](Model_Section_EN.md) | 经过方法和结果复核的英文源稿 |
| [figures/forecast_performance.png](figures/forecast_performance.png) | 整体及全部8场的预测误差 |
| [figures/arrival_risk_calibration.png](figures/arrival_risk_calibration.png) | 30分钟满位概率校准与各箱样本数 |
| [tables/risk_by_carpark.csv](tables/risk_by_carpark.csv) | 按严格共同口径汇总的每场风险样本、事件、Brier与AP |

主要数值来自 [标准两年评估](../../examples/history_comparison/two_year_standard/evaluation/)、[历史长度对比](../../examples/history_comparison/comparison/)、[RQ补充](../../examples/rq_supplement/README.md) 和 [RQ2规则扩展](../rq2_gate/README.md)。实验基准代码版本为 ed35cac。输入数据 SHA-256 见文末。组内提供的数据报告与原始 CSV 没有随本文重复上传。

## 重新生成

```sh
python -m pip install -r requirements-report.txt
python scripts/build_model_section_figures.py
python scripts/build_model_section_docx.py
```

这两条构建命令只读公开结果汇总和源稿。Word 内容更新后应重新渲染检查。页数会随 Word 版本和字体替换略有变化。

重算每场风险表需已保存的本地标准测试预测，以及 requirements-evaluation.txt：

```sh
python scripts/summarize_model_section_risk.py
```

该脚本复用评估的有效出发、已知到达、同段标签和四模型共同概率过滤，并在保存前核对总体样本数、事件数、Brier和AP与公开结果一致。单一类别场的AP留空。完整预测与内部排版检查文件留在 outputs/，不公开上传。
