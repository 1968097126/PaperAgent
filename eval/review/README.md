# 人工标注复核指南

这里保存 Phase 7 的人工复核工作流。`work/` 是本地可再生成、但可能包含审核人信息和未完成意见的工作目录，因此默认不提交；最终经过校验和明确确认的复核记录会在后续独立提交中归档。

当前源数据是 `synthetic_grounded_in_real_corpus`，运行导出或进度检查不会改变来源。人工通过后只能称为“经过人工复核的合成标注”，不能称为“从零人工编写的标注”。

## 1. 生成复核包

在仓库根目录执行：

```bash
conda run -n nanocc python -m eval.review_annotations export \
  --data-dir eval/data \
  --chunks corpus/chunks.jsonl \
  --review-dir eval/review/work
```

命令先运行严格数据校验，然后生成：

- `annotations.csv`：375 条独立复核记录，顺序为 M 120、I 100、R 100、B 55。
- `evidence.csv`：所有 ground-truth chunk 的去重全文目录。
- `review_manifest.json`：五个数据集和语料 chunks 的 SHA-256、数量与 review schema 版本。

E 的 70 条记录来自 M。`annotations.csv` 的 M 行通过 `e2e_id` 标出是否进入 E，因此不要求重复复核 E 行；程序仍会验证 E 与 M 完全一致。

如果目录已经存在，导出会拒绝覆盖，以保护人工填写内容。只有确认旧工作可以丢弃时才使用 `--force`。

## 2. 只能填写的列

不要编辑 `dataset` 到 `record_sha256` 的机器列，也不要编辑 `evidence.csv` 或 `review_manifest.json`。校验器会拒绝任何修改或源数据漂移。

只填写以下六列：

| 列 | 填写规则 |
|---|---|
| `decision` | `approve`、`revise`、`reject`；未审核保持 `pending` |
| `issue_codes` | 多项以英文分号分隔，例如 `intent;retrieval_flag` |
| `proposed_correction_json` | `revise` 时必填 JSON 对象，例如 `{"intent":"follow_up","needs_retrieval":true}` |
| `reviewer` | 非 `pending` 行必填稳定审核人标识；不要填写 API Key 等秘密 |
| `reviewed_at` | 非 `pending` 行必填带时区 ISO 8601，例如 `2026-09-05T16:30:00+08:00` |
| `notes` | 解释拒绝原因、歧义或证据判断；`reject` 时必填 |

允许的 `issue_codes`：

```text
query; history; intent; standalone_query; reference_resolution;
retrieval_flag; ground_truth; expected_action; rewrite_flag;
second_retrieval_flag; refusal_label; comparison_label;
e2e_selection; other
```

`approve` 表示本行全部标签正确，因此不能同时填写 issue 或 correction。`revise` 表示样本可修复，必须给出 issue 和 correction。`reject` 仅用于样本本身不可用，必须给出 issue 和 notes。存在 `revise` 或 `reject` 时，整个复核不能宣告完成。

## 3. 推荐审核顺序与标准

### M：120 条多轮主集

按 `conversation_id`、`turn` 排序，每组连续审核四轮：

1. `query` 与 `history_json` 是否形成自然、无矛盾的上下文。
2. `intent` 是否严格属于 new_search、follow_up、clarification、meta、out_of_scope 中唯一一类。
3. `standalone_query` 是否消除了指代，同时没有添加历史和记忆中不存在的实体或限制。
4. resolved references、required entities/constraints、comparison targets 是否完整且可追溯。
5. `needs_retrieval`、`requires_rewrite`、`requires_second_retrieval` 是否符合问题性质；二次检索只标记首轮证据可能不足、但改写后仍可检索的问题。
6. `expected_action`、`should_refuse`、`refusal_reason` 是否一致。越界问题应拒绝；语料中没有指定结果的 insufficient-evidence 问题也应拒绝。
7. 有 ground truth 时，在 `evidence.csv` 按 chunk ID 查找全文，确认它能直接支持问题；跨论文比较应至少有两个正确来源。
8. `e2e_id` 非空时，额外判断该案例是否适合作为端到端样本；若不适合，使用 `e2e_selection`。

### I：100 条意图集

每类各 20 条。只依据 `query` 与 `history_json` 判断 `intent`，不要因为来源 M 的原标签而跳过独立判断。重点区分 follow_up 与 clarification、meta 与普通论文检索、out_of_scope 与可安全回答的研究问题。

### R：100 条检索集

逐行在 `evidence.csv` 查找每个 ground-truth ID：

1. Query 是否表达了可从该 chunk 检索出的具体信息，而不只是堆叠标题或随机关键词。
2. Chunk 是否直接相关，而非仅来自同一篇论文。
3. 如一个问题合理需要多个证据，修正建议应给出完整 `ground_truth_chunk_ids` 列表。

### B：55 条边界集

仅判断入口动作：论文研究问题为 `allow`；需要用户消除关键歧义为 `clarify`；明显恶意、非法或研究域外请求为 `refuse`。不要把“当前语料证据不足”误标为 Guardrails 拒绝，它属于后续 Reflect 判断。

## 4. 随时检查进度

```bash
conda run -n nanocc python -m eval.review_annotations check \
  --data-dir eval/data \
  --chunks corpus/chunks.jsonl \
  --review-dir eval/review/work
```

命令会输出总进度和各数据集的 pending/approve/revise/reject 数量，并检查机器列、证据目录、manifest、审核时间和 correction JSON。

## 5. 修正与最终确认

如果出现 `revise` 或 `reject`：

1. 不要直接修改机器列或源 JSONL。
2. 保存 `annotations.csv`，将校验命令输出和需要修正的行交回项目维护者。
3. 维护者在独立提交中修正源数据、重新运行严格数据校验并重新导出复核包。
4. 至少重新审核所有受修正影响的 M/I/R/B 行及相关 E 映射。

只有所有 375 行都在修正后成为 `approve`，才运行：

```bash
conda run -n nanocc python -m eval.review_annotations check \
  --data-dir eval/data \
  --chunks corpus/chunks.jsonl \
  --review-dir eval/review/work \
  --require-complete
```

该命令通过后，请明确告知项目维护者“人工复核已完成并同意归档”。维护者随后归档独立的 review status/attestation，但无论复核前后都不修改数据集的 `synthetic`/`provenance` 来源声明。收到人工确认前不得称为 human-reviewed；收到后应称为 human-reviewed synthetic annotations，而不是 human-authored labels。
