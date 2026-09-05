# 最终基线审计

审计日期：2026-09-05。本文冻结 Phase 1–6 的实现边界，并记录进入人工复核和正式评测前的可验证基线。审计结论仅来自仓库代码、离线测试和已有运行记录，不把 mock/fixture 结果表述为正式模型指标。

## 冻结范围

| 层级 | 冻结入口或契约 | 回归证据 |
|---|---|---|
| 语料与索引 | `RetrievalConfig`、稳定 chunk ID、版本化 manifest、FAISS/BM25 构建与 stale 检查 | `tests/test_corpus_index.py`、`tests/test_corpus_edge_cases.py` |
| 检索与工具 | `HybridSearcher`、RRF、Cross-Encoder fallback、`search_papers`、`retrieve_chunk` | `tests/test_retrieval_v2.py`、`tests/test_domain_tools.py` |
| Agent harness | `AgentState`、`run_research()`、研究工具双层白名单、统一 `TurnTrace` | `tests/test_agent.py`、`tests/test_trace.py`、`tests/test_tool_registry.py` |
| 七阶段 Pipeline | Guardrails → NLU → Retrieve → Rerank → Reflect → Generate → Memory；最多一次二次检索 | `tests/test_pipeline_stage1.py` 至 `tests/test_pipeline_stage4.py`、`tests/test_pipeline.py` |
| Provider 与 CLI | 原生 Provider completion 适配、默认研究 REPL、`--coding` 兼容入口、研究状态命令 | `tests/test_provider_updates.py`、`tests/test_research_cli.py` |
| 评测 | M/I/R/B/E 数据契约、覆盖率显式指标、结果持久化、凭据缺失时调用前失败 | `tests/test_eval.py`、`eval.validate_datasets`、dataset-only smoke test |

冻结表示后续阶段不再无依据改变这些入口和语义。若正式评测暴露缺陷，应以独立修复提交修改，并重新执行本页的验收矩阵。

## 遗留项审计

- 运行时 Python 文件未发现 `FIXME`、`NotImplementedError` 或未实现 TODO。演示脚本中的 “TODO comments” 是示例提问，不是代码欠账。
- 空 `pass` 均位于异常类型声明、可选依赖处理、缺失历史文件或配置/上下文读取失败的容错分支；这些路径已有调用方降级语义，不属于占位实现。
- `pipeline/stages.py` 的未提交差异是格式化结果；模块说明已校正为覆盖完整的可独立测试阶段。
- `tests/test_pipeline.py` 与细粒度阶段测试部分重叠，但保留为低成本的跨阶段回归 smoke suite，防止单组件测试通过而组合契约漂移。
- `docs/superpowers/` 中的历史规格与计划只作为设计记录，不是运行时完成状态的权威来源；当前状态以 `docs/implementation_status.md` 为准。

## 一致性结论

1. Agent 层的研究状态字段能被 Pipeline 原子更新，并由 CLI 保存、加载、清理和展示。
2. Pipeline 的检索阶段复用 Phase 2 搜索器；Provider 只参与语义、反思和生成，不获得研究工具执行权。
3. 正常、拒绝、澄清、无需检索和异常降级路径都保留七阶段 Trace 结构；实际执行的检索尝试另以 attempt 元数据保存完整血缘。
4. 最终引用仅接受当前证据集合中的 `[paper_id/chunk_id]`，进入状态和 Memory 的引用经过同一校验。
5. 评测数据保持 `synthetic_grounded_in_real_corpus` 标记；在人工逐条确认前不得改称 human-labelled。
6. Provider 评测在缺少所选模型凭据时于发起模型请求前失败；dataset suite 与离线回归不依赖 API Key。

## 阶段 1 验收矩阵

```bash
conda run -n nanocc python -m pytest -q tests/test_pipeline.py
conda run -n nanocc python -m pytest -q
conda run -n nanocc ruff check agent.py pipeline research_cli.py eval tests
conda run -n nanocc mypy --follow-imports=skip agent.py pipeline research_cli.py eval
conda run -n nanocc python -m eval.validate_datasets \
  --data-dir eval/data --chunks corpus/chunks.jsonl
conda run -n nanocc python -m eval.run_evaluation \
  --suite datasets --seed 42 --output-dir /tmp/nanocc-final-audit
git diff --check
```

还应执行只读扫描，确认受版本控制的文件中没有可疑凭据或本机绝对路径。扫描只报告命中位置，不输出可能的凭据值。

## 静态检查边界

- Ruff 对 Agent、Pipeline、CLI、评测代码和全部测试执行；本阶段同时清理了继承测试中的导入排序、未使用导入和未使用局部变量。
- mypy 对 Phase 1–6 新增的 Agent、Pipeline、CLI 和评测模块执行，使用 `--follow-imports=skip` 将检查边界固定在这些显式目标。
- 若让 mypy 递归进入继承模块，当前会在 `providers.py`、`tools.py`、`retrieval/dense_index.py` 和 `retrieval/sparse_index.py` 报告 38 个既有问题，主要来自第三方 SDK 联合类型、旧式 Optional 注解和由 v2 实现覆盖的兼容导出。它们未导致本阶段回归失败，也未在代码冻结时做高风险批量改写；若发布标准要求全仓 mypy 零告警，应先独立清理并重新运行验收矩阵。

## 尚未完成的发布门槛

1. 人工复核 M/I/R/B/E 标注，并留下可追溯的复核记录。
2. 在固定 Provider、模型版本、配置和随机种子后执行检索、消融及 Provider-backed 正式评测。
3. 将正式结果、限制、复现实验命令和代表性真实 Trace 写入最终 README/报告。

以上三项不属于本阶段完成声明，也不影响 Phase 1–6 离线基线的可复现性。
