# 实施状态

最后更新：2026-09-04。状态只依据已提交或准备提交的代码及实际执行记录；fixture/mock 结果不作为正式指标。

| Phase | 状态 | 代码证据 | 测试/运行证据 |
|---|---|---|---|
| 1 语料库与索引 | 已完成 | `retrieval/config.py`、`retrieval/corpus.py`、`retrieval/chunking_v2.py`、`retrieval/dense_v2.py`、`retrieval/sparse_v2.py`、`retrieval/manifest.py`、`corpus/indexing.py` | `tests/test_corpus_index.py`、`tests/test_corpus_edge_cases.py`；真实建库生成 30 papers / 1,800 chunks / 1,800×1024 FAISS 向量，manifest `ready=true, stale=false` |
| 2 混合检索与重排序 | 已完成 | `retrieval/fuse_v2.py`、`retrieval/rerank_v2.py`、`retrieval/hybrid_v2.py`、`domain_tools.py`、`tool_validation.py`、`retrieval/sanity_check.py` | `tests/test_retrieval_v2.py`、`tests/test_domain_tools.py`；三档真实检索输出 `docs/retrieval_sanity.json` |
| 3 Harness 内核改造 | 已完成 | `agent.py`、`pipeline/trace.py`、`tool_registry.py`；保留 Provider、权限门、上下文压缩和原编程入口 | `tests/test_agent.py`、`tests/test_trace.py`、`tests/test_tool_registry.py`、`tests/test_domain_tools.py`；研究域 schema/执行双层白名单、允许/禁止/未知工具、非法参数、Provider 异常和兼容路径均有测试 |
| 4 七阶段编排 | 未完成 | — | 尚未完成收口和真实 Provider 端到端验证 |
| 5 CLI | 未完成 | — | 尚未完成收口和真实 Provider CLI 验证 |
| 6 评测系统 | 未完成 | — | 尚无正式全量评测结果 |
| 7 打磨与真实结果 | 未开始 | — | — |

## 当前可复现命令

```bash
conda run -n nanocc python corpus/build_index.py --corpus-dir corpus --device cpu
conda run -n nanocc python -m retrieval.sanity_check --corpus-dir corpus --device cpu --top-k 3 --output docs/retrieval_sanity.json
conda run -n nanocc python -m pytest -q
conda run -n nanocc python -m pytest -q tests/test_agent.py tests/test_trace.py tests/test_tool_registry.py tests/test_domain_tools.py
```

## Phase 1–2 已验证事实

- 语料来源为仓库本地的 30 篇论文正文；`tests/fixtures/corpus/` 只用于单元测试。
- chunk 策略支持 `sliding` 与 `paragraph`，overlap、最短长度均可配置；ID 由论文、章节、文本内容的哈希稳定生成。
- 索引格式版本为 2；manifest 保存语料指纹，语料变化会被标记为 stale，未变化时跳过重建。
- 检索结果在相同索引和配置下使用显式 tie-break，RRF 会去除单路重复 chunk。
- Cross-Encoder 异常采用保留融合顺序的显式 fallback；工具异常统一返回 `Error: ...`。

## Phase 3 已验证事实

- `AgentState` 在原消息与 token 状态之上增加语义状态、检索状态、引用、研究记忆、记忆模式和逐轮 Trace。
- 原 `run(..., tool_names=None)` 行为保留，供继承的编程 Agent 使用；`run_research()` 固定为 `search_papers`、`retrieve_chunk` 两个只读领域工具。
- 研究工具范围同时约束发给 Provider 的 schema 和工具执行入口，模型猜测出的白名单外工具不会进入分发器。
- 未知工具与非法参数继续使用稳定的 `Error: ...` 字符串，不导致 Agent 循环崩溃。
- Turn/Stage Trace 记录状态、错误、起止时间、耗时、输入/输出摘要和元数据；Provider 异常会先完成错误 Trace 再向调用方抛出。
- Trace 与消息历史分离，不受 nano-cc 原有上下文压缩裁剪影响。
- 工具注册表测试会恢复进程级全局状态，测试结果不依赖执行顺序。

## 当前外部阻塞

`nanocc` 当前默认 Provider 为 Anthropic，未配置 API Key。因此 Phase 4/5 的真实模型端到端验证和 Phase 6 的模型相关正式评测暂时不能执行。该阻塞不影响离线索引、检索、fixture/mock 测试、纯指标测试及后续离线实现。
