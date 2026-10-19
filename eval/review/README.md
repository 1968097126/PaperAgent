# 人工标注复核

在索引和评测数据就绪后，从项目根目录导出复核表：

```bash
python -m eval.review_annotations export --review-dir eval/review/work
```

1. 阅读导出的题目、对话历史和证据，在审核列填写 `approve`、`revise` 或 `reject`；未处理行保持 `pending`。
2. 非 `pending` 行填写 `reviewer` 和含时区的 `reviewed_at`；需修改或拒绝的条目填写问题类型与说明。
3. 保留机器列、证据文件及校验清单原样，避免源数据与复核内容脱节。复核工具只记录意见，不自动改变标准答案。

检查复核表：

```bash
python -m eval.review_annotations check --review-dir eval/review/work
# 全部处理后再执行完整性检查
python -m eval.review_annotations check --review-dir eval/review/work --require-complete
```

复核表为本地产物，不提交 Git。通过完整性校验不等于答案已由程序证明正确；人工复核状态应如实记录，模型辅助生成的来源声明始终保留。
