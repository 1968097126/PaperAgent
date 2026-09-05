# 实施状态

最后更新：2026-09-05。状态只依据已提交或准备提交的代码及实际执行记录；fixture/mock 结果不作为正式指标。

| Phase | 状态 | 代码证据 | 测试/运行证据 |
|---|---|---|---|
| 1 语料库与索引 | 已完成 | `retrieval/config.py`、`retrieval/corpus.py`、`retrieval/chunking_v2.py`、`retrieval/dense_v2.py`、`retrieval/sparse_v2.py`、`retrieval/manifest.py`、`corpus/indexing.py` | `tests/test_corpus_index.py`、`tests/test_corpus_edge_cases.py`；真实建库生成 30 papers / 1,800 chunks / 1,800×1024 FAISS 向量，manifest `ready=true, stale=false` |
| 2 混合检索与重排序 | 已完成 | `retrieval/fuse_v2.py`、`retrieval/rerank_v2.py`、`retrieval/hybrid_v2.py`、`domain_tools.py`、`tool_validation.py`、`retrieval/sanity_check.py` | `tests/test_retrieval_v2.py`、`tests/test_domain_tools.py`；三档真实检索输出 `docs/retrieval_sanity.json` |
| 3 Harness 内核改造 | 已完成 | `agent.py`、`pipeline/trace.py`、`tool_registry.py`；保留 Provider、权限门、上下文压缩和原编程入口 | `tests/test_agent.py`、`tests/test_trace.py`、`tests/test_tool_registry.py`、`tests/test_domain_tools.py`；研究域 schema/执行双层白名单、允许/禁止/未知工具、非法参数、Provider 异常和兼容路径均有测试 |
| 4 七阶段编排 | 已完成 | `pipeline/stages.py`、`pipeline/memory.py`、`pipeline/orchestrator.py`、`pipeline/client.py`、`providers.py` | mock/异常/真实索引测试已通过；DeepSeek V4 Flash 真实七阶段回答与 OpenAI GPT-5.6 Luna completion 已验证 |
| 5 CLI | 已完成 | `nano_claude.py`、`research_cli.py` | `tests/test_research_cli.py`；mock Provider 完整跑通一轮七阶段，真实语料 `/corpus` smoke test 已验证 |
| 6 评测系统 | 已完成 | `eval/datasets.py`、`eval/metrics.py`、`eval/run_evaluation.py`、`eval/validate_datasets.py`、`eval/data/*.jsonl` | `tests/test_eval.py`；真实语料绑定的 M/I/R/B/E 严格校验通过，正式 Provider 指标留待 Phase 7 |
| 7 打磨与真实结果 | 进行中：收尾阶段 1 已完成 | `docs/final_audit.md`、`tests/test_pipeline.py`；冻结 Phase 1–6 的运行时接口，不提前写入正式评测结果 | 全量回归、Phase 1–6 目标模块静态检查、数据集校验与离线 smoke test 均通过；人工复核和正式 Provider 评测留待后续收尾阶段 |

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
conda run -n nanocc python -m pytest -q tests/test_pipeline.py
conda run -n nanocc python -m pytest -q tests/test_provider_updates.py
conda run -n nanocc python -m pytest -q tests/test_research_cli.py
conda run -n nanocc python nano_claude.py --help
conda run -n nanocc python -m pytest -q tests/test_eval.py
conda run -n nanocc python -m eval.validate_datasets --data-dir eval/data --chunks corpus/chunks.jsonl
conda run -n nanocc python -m eval.run_evaluation --suite datasets --build-datasets --seed 42
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
- DeepSeek V4 Flash 已在 30 篇论文真实索引上完成一轮充分证据回答：七阶段状态均为 `ok`，4 个引用全部来自最终证据并写入 Memory。
- OpenAI GPT-5.6 Luna 已完成真实 completion 与 token usage 验证；GPT-5/6 使用 `max_completion_tokens`，流式请求显式开启 usage 回传。
- DeepSeek V4 默认高强度思考可能耗尽结构化阶段的输出预算；Provider 现在默认显式关闭思考，只有配置 `thinking=true` 时才按 low/high/max 开启。

## 当前外部配置

真实验证使用了用户临时提供的 OpenAI/DeepSeek 凭据；仓库已恢复为 `OPENAI_API_KEY` 与 `DEEPSEEK_API_KEY` 环境变量引用，没有保留或提交任何密钥。后续真实 CLI 或正式评测前需要重新导出相应环境变量。

## Phase 5 已验证事实

- `nano_claude.py` 默认进入科研论文问答模式，并保留 `--coding` 作为继承编程 Agent 的显式离线开发入口；研究模式不进入工具 diff 等编程展示链路。
- `/trace [turn_id]` 支持最新轮次、轮次序号和稳定 turn ID，输出七阶段详情；空 Trace、不存在轮次和损坏 Trace 返回稳定错误对象。
- `/corpus` 汇总论文数、chunk 数、索引 ready/stale/缺失状态、索引文件、版本、Dense/Reranker 模型和 device；空语料、部分索引和损坏 chunks 均有测试。
- `/memory [off|on|compressed]` 展示结构化记忆、估算 token、更新次数、压缩次数与当前压缩状态，并支持三态切换。
- `/clear` 同步清空研究状态；`/save` 与 `/load` 持久化并恢复语义、检索、引用和 Memory 状态；`/config` 的 Provider Key 在列表、查询和设置回显中均被遮蔽。
- mock Provider 集成测试从 CLI 入口跑通 Guardrails → NLU → Retrieve → Rerank → Reflect → Generate → Memory，并断言 CLI 未重复累计 Orchestrator 已记录的 token。
- 按用户要求，本步骤不重新进行付费 Provider CLI 调用；Phase 4 已完成 DeepSeek V4 与 OpenAI Provider 的真实接口验证，后续可在重新导出环境变量后人工执行 CLI 验收。

## Phase 6 已验证事实

- M/I/R/B/E 共 445 行物理记录；E 的 70 行严格派生自 M，因此独立范围统计不重复相加。全部记录明确标记为 `synthetic_grounded_in_real_corpus`，不得称为人工标注。
- M 严格包含 30 组四轮对话及指定五类意图分布；46 个改写、24 个二次检索、24 个拒答和 12 个跨论文比较是允许重叠的属性。
- 所有检索 ground truth 都是当前 1,800 个真实 chunk 之一；6 个证据不足问题使用唯一表格标记，验证器确认标记未出现在语料正文中。
- 验证器覆盖字段类型、稳定/唯一 ID、对话轮次、精确分布、E⊂M 内容一致性、引用存在性、比较对象数量及拒答标签一致性。
- 指标覆盖 Intent confusion matrix、非全文匹配的 Query 改写保留率、三档 Recall@1/5/10、Guardrails、Reflect on/off、引用 validity/precision/recall、Pipeline 合取成功率、Memory 三态和 token/context 开销。
- 每项指标分别报告 evaluated、failed、skipped 和 unevaluable；逐样本与汇总输出支持 JSONL、JSON、CSV 和 Markdown，运行配置记录 seed、模型、Provider、索引 manifest、检索配置及数据来源。
- `datasets` suite 不需要索引 manifest 或 API Key；`provider`/`all` 在凭据缺失时会在模型调用前明确失败。Phase 6 不发布百分比，真实全量评测和人工复核属于 Phase 7。

## Phase 7 收尾阶段 1 已验证事实

- 已审计运行时代码中的 TODO、FIXME、`NotImplementedError` 和空实现；唯一 TODO 命中是继承演示文案，`pass` 命中均属于异常容错或异常类型声明，不是功能占位。
- `pipeline/stages.py` 的遗留内容确认为 Ruff 机械格式化，并恢复了覆盖全部七阶段的准确模块说明；没有改变阶段行为。
- 原未跟踪的 `tests/test_pipeline.py` 已整理为最终七阶段回归 smoke suite，覆盖结构化修复、安全 fallback、阶段顺序、二次检索硬上限、引用校验和 Memory 三态。
- Phase 1–6 的公共入口、七阶段顺序、Provider 适配、状态更新、数据集校验与 CLI 路径均纳入最终审计矩阵；结论与剩余门槛见 `docs/final_audit.md`。
- 本阶段不调用付费 Provider、不修改评测标注来源，也不发布正式百分比；人工复核、正式评测和发布文档仍按后续阶段独立提交。
