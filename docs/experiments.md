# 实验说明（2026-10-19）

## 配置与数据

1. 重新选择 20 篇 Agent / LLM 论文：2 篇期刊、18 篇会议论文，正式发表时间在 2023-10-19 至 2026-10-19 内。会议包括 ICLR、ICML、NeurIPS、ACL、EMNLP，期刊为 TOIS、JMLR；NeurIPS 基准轨道单独注明。来源见 [论文清单](../corpus/selected_papers.md)。
2. 保存 721 页全文、提取文本及 SHA-256；建立 1,759 个分块和 1,024 维 FAISS 索引。Memory Survey 使用固定版本的作者预印本，期刊信息另存，未将其冒充期刊最终版。
3. 环境为 Python 3.11，虚拟环境 `.venvs/paperagent`。模型为 `deepseek-flash`，接口为 `https://api.deepseek.com`；BGE-M3 和重排序模型固定到具体版本。依赖见 `requirements.lock.txt`，密钥仅保存在被忽略的 `.env`。
4. 重建 30 组四轮对话及 I100、R100、B55、E70，完成组件、检索、Reflect 和三种记忆模式实验。1,021 条评测记录中 1,020 条正常、1 条 NLU 解析失败；E70 复用真实连续对话结果，不代表额外 70 次独立调用。
5. 198 项测试通过，真实 CLI 问答已验证。模型版本与数据校验值保存在结果快照中。

## 本次结果

| 项目 | 结果 | 口径 |
| --- | --- | --- |
| 意图五分类 | 80/100，80.0% | 五类各 20 条 |
| Query 改写 | 39/46，84.8% | 必需信息保留；1 条解析失败计入分母 |
| 边界路由 | 49/55，89.1% | allow / clarify / refuse，不等同于拒答准确率 |
| 纯稠密 Recall@5 | 57/100，57.0% | 每条只有一个标注相关块 |
| 混合检索 Recall@5 | 32/100，32.0% | BGE-M3 + BM25 + RRF |
| 混合加重排序 Recall@5 | 68/100，68.0% | 比纯稠密高 11 个百分点 |
| E70 Pipeline Success | 40/70，57.1% | 路由、检索、action、引用和 trace 联合判定 |
| 压缩 token 节省 | 1.83% | 未压缩 442,650 → 压缩 434,538，总输入输出 tokens |
| 记忆字段保留率 | 未压缩 80.7%；压缩 75.9% | 仅统计有必需字段的 88 轮，不等同语义一致性 |

引用格式有效率为 100%；精确命中标注块的引用 Precision / Recall 为 19.1% / 44.0%。标注未穷尽所有可支持答案的段落，因此两者均不能直接叫“引用事实准确率”。另用 DeepSeek 对 41 个带引用回答、335 项事实做自动支持性审计，全部判为支持；这是同模型系列审计，尚无人工验证，不能作为 100% 正确的结论。

## Reflect 消融

固定标注 Query，并仅在组件消融中绕过 Guardrails / NLU，以避免其他阶段干扰。24 条可回答问题首轮人为移除整篇目标论文的候选，另有 6 条询问不存在表格的问题；允许一次重检时恢复正常候选。此实验反映受控压力下的恢复能力，不是自然流量表现。

| 条件 | 可回答问题未拒答 | 证据不足时拒答 | 总拒答决策正确率 |
| --- | --- | --- | --- |
| Reflect 关闭 | 5/24 | 4/6 | 9/30，30.0% |
| Reflect 开启、不重检 | 0/24 | 6/6 | 6/30，20.0% |
| Reflect 开启、一次重检 | 24/24 | 5/6 | 29/30，96.7% |

上表按实际回答文本由模型盲审，待人工复核。按结构化 action 计分则分别为 40.0%、20.0%、93.3%，差异来自“文字已经拒答，action 仍为 answer”。开启重检后仍有 1 条这种不一致。证据不足子集仅改善 1 个样本，不能据此宣称稳定提升。

## 结果与限制

1. 原始结果、逐项调用、耗时、token 用量和失败信息在 `eval/results/rebuild-20261019`；小型结果快照为 [results.json](results.json)。原始运行产物保留在本地，不随 Git 仓库分发。
2. 题库为真实全文支持的模型辅助标注，已检查摘录在原文中出现，但人工语义复核未完成。I/R/B/E 有关联，并且开发时观察过这些样本，不能当作独立泛化成绩。可按 [复核说明](../eval/review/README.md) 导出人工审核材料。
3. 当前混合召回退步，重排序才带来最终收益。后续应检查中英文词项匹配、RRF 权重和多相关块标注；尚未确认退步原因。
4. 当前记忆实验只有四轮；61 轮发生在压缩之后，实际 token 收益较小。应使用更长对话进一步检验，目前不足以证明长期对话中的压缩收益。
5. 优先后续工作是人工复核失败样例、统一生成文本与 action、优化混合召回。本表仅描述当前配置下的实验表现。

## 复现

在项目根目录执行，先在 `.env` 中配置 `DEEPSEEK_API_KEY`。现有题库和检查点会复用；需要独立重跑时给运行器指定新的 `--output-dir`。

```bash
source .venvs/paperagent/bin/activate
python -m pip install -r requirements.lock.txt
python -m corpus.acquire_selected
HF_HUB_DISABLE_XET=1 python -m corpus.build_index --cache-dir .cache/models --device mps --batch-size 8
python -m eval.build_research_data
python -m eval.run_rebuild --suite all --device mps
python -m eval.judge_citations
python -m eval.audit_refusals
python -m pytest -q
python scripts/export_experiment.py
```

没有 Metal 的机器将 `mps` 改成 `cpu`。下载的模型位于 `.cache/models`。备份包含全文、索引、题库、代码和实验结果，不含密钥和模型权重；归档路径见 `artifacts/`，建议另存一份到自己的备份位置。
