# 实施状态

最后更新：2026-09-05。状态只依据已提交或准备提交的代码及实际执行记录；fixture/mock 结果不作为正式指标。

| Phase | 状态 | 代码证据 | 测试/运行证据 |
|---|---|---|---|
| 1 语料库与索引 | 已完成 | `retrieval/config.py`、`retrieval/corpus.py`、`retrieval/chunking_v2.py`、`retrieval/dense_v2.py`、`retrieval/sparse_v2.py`、`retrieval/manifest.py`、`corpus/indexing.py` | `tests/test_corpus_index.py`、`tests/test_corpus_edge_cases.py`；真实建库生成 30 papers / 1,800 chunks / 1,800×1024 FAISS 向量，manifest `ready=true, stale=false` |
| 2 混合检索与重排序 | 已完成 | `retrieval/fuse_v2.py`、`retrieval/rerank_v2.py`、`retrieval/hybrid_v2.py`、`domain_tools.py`、`tool_validation.py`、`retrieval/sanity_check.py` | `tests/test_retrieval_v2.py`、`tests/test_domain_tools.py`；三档真实检索输出 `docs/retrieval_sanity.json` |
| 3 Harness 内核改造 | 已完成 | `agent.py`、`pipeline/trace.py`、`tool_registry.py`；保留 Provider、权限门、上下文压缩和原编程入口 | `tests/test_agent.py`、`tests/test_trace.py`、`tests/test_tool_registry.py`、`tests/test_domain_tools.py`；研究域 schema/执行双层白名单、允许/禁止/未知工具、非法参数、Provider 异常和兼容路径均有测试 |
| 4 七阶段编排 | 已完成（真实 Provider 验证待人工补充） | `pipeline/stages.py`、`pipeline/memory.py`、`pipeline/orchestrator.py`、`pipeline/client.py` | `tests/test_pipeline_stage1.py` 至 `tests/test_pipeline_stage4.py`；mock E2E、异常降级和微型真实索引 smoke test 已通过 |
| 5 CLI | 未完成 | — | 尚未完成收口和真实 Provider CLI 验证 |
| 6 评测系统 | 未完成 | — | 尚无正式全量评测结果 |
| 7 打磨与真实结果 | 未开始 | — | — |

## 当前可复现命令

```bash
conda run -n nanocc python corpus/build_index.py --corpus-dir corpus --device cpu
conda run -n nanocc python -m retrieval.sanity_check --corpus-dir corpus --device cpu --top-k 3 --output docs/retrieval_sanity.json
conda run -n nanocc python -m pytest -q
conda run -n nanocc python -m pytest -q tests/test_agent.py tests/test_trace.py tests/test_tool_registry.py tests/test_domain_tools.py
conda run -n nanocc python -m pytest -q tests/test_pipeline_stage1.py
conda run -n nanocc python -m pytest -q tests/test_pipeline_stage2.py
conda run -n nanocc python -m pytest -q tests/test_pipeline_stage3.py
conda run -n nanocc python -m pytest -q tests/test_pipeline_stage4.py
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

## Phase 4 Step 1 已验证事实

- Guardrails 使用确定性规则处理空输入、超长输入和明显提示注入，再用可选 Provider 做粗粒度领域判断。
- Guardrails 负责明显非法、注入和粗粒度越界；NLU 保留 `out_of_scope` 作为语义兜底，两者边界已写入模型提示。
- NLU 固定输出五类意图、独立 Query、已解析引用、检索决策、原因、约束和比较对象，并严格校验字段类型与意图/检索决策一致性。
- 模型结构化输出只允许一次修复；两次解析失败转为澄清，Provider 异常转为带尝试次数的安全 fallback。
- 已解析引用、约束和比较对象必须来自当前输入、最近历史或已验证记忆，并原样保留在独立 Query 中；不可追溯值会触发修复或澄清。
- 无 Provider 时不伪造语义分类：Guardrails 延迟给 NLU，NLU 使用原始 Query 的显式 fallback。

## Phase 4 Step 2 已验证事实

- Retrieve 与 Rerank 只委托 Phase 2 `HybridSearcher.candidates()` 和 `rerank_candidates()`，编排层不复制稠密检索、BM25、RRF 或 Cross-Encoder 算法。
- 候选与证据输出校验非空 `chunk_id`、重复 ID 及重排结果来源；重排层不能引入召回候选之外的 chunk。
- Reflect 严格校验 `sufficient`、`missing_information`、四类 action、`rewritten_query` 和 reason 的类型、完整性与相互一致性。
- Reflect 结构化输出只允许一次修复；持续解析失败或 Provider 异常都转为可解释的安全拒答。
- `EvidenceRefiner` 默认最多执行一次二次检索，两轮总上限由 `max_second_retrievals` 显式控制；模型在预算耗尽后继续请求检索会被策略层改为拒答。
- 每次检索保留 attempt、Query、完整候选、最终证据、候选/证据 ID、Reflect 决策和错误状态；传入 `TurnTrace` 时同步记录 Retrieve/Rerank/Reflect 的起止时间、耗时、输入输出和 attempt 元数据。
- 空结果在预算内做确定性重试，预算耗尽后拒答；检索或重排异常不会被伪装成正常空结果。

## Phase 4 Step 3 已验证事实

- Generate 只在 Reflect 明确 `sufficient=true, action=answer` 且证据结构完整时进入生成；Meta、澄清、越界与证据不足均走不调用 Provider 的确定性响应。
- 生成提示只提供最终证据，中文提问默认中文，并限制每条证据进入上下文的长度；无 Provider 时返回带真实 `[paper_id/chunk_id]` 的证据摘录 fallback。
- 引用校验程序化覆盖缺失引用、未知引用、重复引用、空证据、畸形证据和重复证据身份；只有与最终证据匹配的 chunk ID 才进入状态与 Memory。
- 模型答案引用非法时只允许一次修复；首次调用异常、修复调用异常或二次校验失败均返回不泄漏未验证答案的安全拒答。
- `ResearchMemory` 支持 `off`、`on`、`compressed` 三态；`compressed` 仅在估算 token 超过阈值时自动压缩，不再每轮无条件压缩。
- Memory 只保存实际引用证据对应的论文、chunk 和事实；实体、限制条件及比较对象必须能从当前输入、显式历史上下文、既有记忆或检索证据中追溯。
- 压缩独立于 nano-cc 的消息级 `compaction.py`：前者约束结构化研究事实，后者负责通用消息上下文；所有 Memory 列表均有压缩上限或硬上限。

## Phase 4 Step 4 已验证事实

- `ResearchPipeline` 统一执行七阶段；无需检索和安全拒绝路径也会以 `skipped` 或确定性决策补齐完整阶段 Trace。
- Orchestrator 直接复用 `EvidenceRefiner`，不复制检索算法或二次检索循环；每次尝试的 Query、候选、证据、决策和错误均写入 `retrieval_state`。
- 检索、重排、语义阶段、生成及损坏 Memory 状态都能降级为安全回答，并在 Trace 中区分 `error`、`degraded` 与 `skipped`。
- 每轮完成后原子式更新消息、语义状态、检索状态、引用、研究记忆、轮次和 Trace；Provider token 使用调用前后差值，避免复用客户端时重复累计。
- `NativeProviderClient` 将原流式 Provider 适配为阶段 completion，显式传入空工具 schema，并累计每次完成的输入/输出 token。
- mock E2E 覆盖标准链路、二次检索上限、多轮历史、安全路由及异常降级；微型真实索引 smoke test 实际构建并加载版本化 FAISS、BM25、chunks 与 manifest，只用确定性替身隔离外部嵌入和重排模型。
- 用户决定暂缓真实 Provider E2E；该验证不是本次 Step 4 代码收口条件，后续取得 API Key 或可用本地模型后人工补充。

## 当前外部阻塞

`nanocc` 当前默认 Provider 为 Anthropic，未配置 API Key，且本机 Ollama/LM Studio 未响应。因此真实 Provider E2E、Phase 5 真实 CLI 验证和 Phase 6 模型相关正式评测待人工补充；Phase 4 的代码、mock E2E 和离线真实索引验证已经收口。
