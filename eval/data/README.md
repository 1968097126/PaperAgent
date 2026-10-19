# 评测数据

| 文件 | 数量 | 用途 |
| --- | --- | --- |
| M.jsonl | 120 turns | 30 组四轮主对话 |
| I.jsonl | 100 | 五类意图识别 |
| R.jsonl | 100 | 检索命中率 |
| B.jsonl | 55 | 边界路由 |
| E.jsonl | 70 turns | 从主对话选出的端到端计分轮次 |

1. 数据由真实全文支持的模型辅助问题构建，人工语义复核尚未完成；各子集有关联，不能当作独立泛化测试集。
2. `python -m eval.validate_datasets` 检查数据结构及证据 ID，需要先建立论文索引。
3. `python -m eval.build_research_data` 从 `eval/question_bank/` 中的固定题库重建数据。删除题库后再运行会重新调用模型，产生费用及不同的问题。
4. 生成或校验人工复核表的方法见 [复核说明](../review/README.md)。实验口径见 [实验说明](../../docs/experiments.md)。
