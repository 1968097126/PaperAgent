"""Build source-checked questions and coherent dialogues for the frozen 20-paper corpus.

Questions are model-assisted drafts, not human-reviewed gold labels. Every factual
answer must include a verbatim support span from its selected source chunk.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.datasets import (
    PROVENANCE,
    build_boundary_dataset,
    build_e2e_dataset,
    build_intent_dataset,
    read_jsonl,
    validate_datasets,
    write_jsonl,
)
from local_env import load_local_env
from pipeline.client import NativeProviderClient


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def question_bank(paper: dict, chunks: list[dict], output: Path) -> list[dict]:
    path = output / f"{paper['paper_id']}.json"
    if path.exists():
        return json.loads(path.read_text())
    usable = [c for c in chunks if len(c["text"]) >= 650 and c["section"] != "Unknown"]
    # Spread candidates over the main text; avoid generating five paraphrases of one sentence.
    candidates = [usable[int(i * (len(usable) - 1) / 19)] for i in range(20)]
    client = NativeProviderClient(
        {
            "model": "deepseek-flash",
            "max_tokens": 4096,
            "temperature": 0,
            "thinking": False,
        }
    )
    prompt = (
        "Create exactly 5 different factual research questions in Chinese for this paper. "
        "The first question should concern the central contribution, other questions concern "
        "methods, evaluation setup, results or limitations. Use natural questions, never concatenate "
        "keywords. Each question must name the paper by its short name. Answer using ONLY ONE "
        'supplied chunk per question. Return JSON {"questions":[{"query":string,"answer":string,'
        '"chunk_id":string,"support_quote":string}]}. support_quote must be an EXACT contiguous '
        "English quote of 60-200 characters from that chunk and must support the answer. "
        "No outside knowledge, no invented numbers, no questions about bibliography. "
        "Use five different chunks. Paper: "
        + paper["title"]
        + "\nChunks:\n"
        + json.dumps(
            [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in candidates],
            ensure_ascii=False,
        )
    )
    by_id = {c["chunk_id"]: c for c in candidates}
    for attempt in range(3):
        raw = client.complete("DatasetAuthor", prompt)
        try:
            payload = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()))
            questions = payload["questions"]
            assert len(questions) == 5
            assert len({q["chunk_id"] for q in questions}) == 5
            for q in questions:
                source = by_id[q["chunk_id"]]
                assert len(q["support_quote"]) >= 40
                assert normalize(q["support_quote"]) in normalize(source["text"])
                q.update(
                    paper_id=paper["paper_id"],
                    title=paper["title"],
                    ground_truth_chunk_ids=[q["chunk_id"]],
                    annotation_status="model_assisted_source_span_checked_pending_human_review",
                )
            path.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n")
            print(f"questions: {paper['paper_id']}", flush=True)
            return questions
        except (ValueError, KeyError, AssertionError):
            prompt += "\nPrevious draft failed validation. Check five distinct IDs and copy support_quote exactly."
    raise ValueError(f"Invalid source spans: {paper['paper_id']}")


def build_dialogues(bank: dict[str, list[dict]]) -> list[dict]:
    groups = list(bank.values())
    result = []
    clarification_count = 0
    out_count = 0
    for i in range(30):
        qs = groups[i % 20]
        other = groups[(i + 1) % 20]
        # Keep the declared intent counts while keeping follow-ups on the same paper.
        # Explicit comparisons introduce the second paper before any ambiguous turn.
        if i < 10:
            intents = ["new_search", "follow_up", "follow_up", "out_of_scope"]
        elif i < 20:
            intents = ["new_search", "follow_up", "new_search", "clarification"]
        elif i < 28:
            intents = [
                "new_search",
                "follow_up",
                "out_of_scope",
                "meta" if i < 26 else "clarification",
            ]
        else:
            intents = ["new_search", "follow_up", "new_search", "clarification"]
        history = []
        for j, intent in enumerate(intents):
            q = qs[j % 5]
            row = {
                "id": f"M-{len(result) + 1:03}",
                "conversation_id": f"M-C{i + 1:02}",
                "turn": j + 1,
                "query": q["query"],
                "history": list(history),
                "intent": intent,
                "standalone_query": q["query"],
                "resolved_references": [],
                "required_entities": [],
                "required_constraints": [],
                "comparison_targets": [],
                "needs_retrieval": intent in {"new_search", "follow_up"},
                "ground_truth_chunk_ids": [],
                "requires_rewrite": intent == "follow_up",
                "requires_second_retrieval": False,
                "should_refuse": False,
                "refusal_reason": "",
                "expected_action": "answer",
                "cross_paper_comparison": False,
                "unanswerable_marker": "",
                "synthetic": True,
                "provenance": PROVENANCE,
                "reference_answer": q["answer"],
                "support_quote": q["support_quote"],
                "annotation_status": q["annotation_status"],
            }
            if intent == "new_search" and j == 0:
                row["ground_truth_chunk_ids"] = q["ground_truth_chunk_ids"]
            elif intent == "new_search":
                row.update(
                    query=f"比较《{qs[0]['title']}》与《{other[0]['title']}》各自的核心贡献。",
                    comparison_targets=[qs[0]["title"], other[0]["title"]],
                    required_entities=[qs[0]["title"], other[0]["title"]],
                    ground_truth_chunk_ids=[qs[0]["chunk_id"], other[0]["chunk_id"]],
                    cross_paper_comparison=True,
                    reference_answer=qs[0]["answer"] + " " + other[0]["answer"],
                )
            elif intent == "follow_up":
                row.update(
                    query=f"继续讨论这篇论文：{q['query']}",
                    required_entities=[q["title"]],
                    resolved_references=[q["title"]],
                    ground_truth_chunk_ids=q["ground_truth_chunk_ids"],
                )
                # Remove the paper's short/full name, retaining an unambiguous topic cue.
                short = q["title"].split(":")[0]
                row["query"] = (
                    row["query"].replace(short, "该方法").replace(q["title"], "该论文")
                )
                row["standalone_query"] = f"关于《{q['title']}》：{q['query']}"
                if i < 6 and j == 2:
                    marker = f"Z-{i + 1}99"
                    row.update(
                        query=f"它在 Table {marker} 中给出的准确率是多少？",
                        standalone_query=f"《{q['title']}》在 Table {marker} 中给出的准确率是多少？",
                        ground_truth_chunk_ids=[],
                        should_refuse=True,
                        refusal_reason="insufficient_evidence",
                        expected_action="refuse",
                        unanswerable_marker=marker,
                        reference_answer="论文中没有该表，证据不足。",
                    )
            elif intent == "clarification":
                clarification_count += 1
                row.update(
                    query=f"第 {i + 1} 组讨论里，那个更好吗？",
                    expected_action="clarify",
                    reference_answer="需要澄清比较对象和指标。",
                )
                if clarification_count <= 6:
                    title = q["title"]
                    row.update(
                        requires_rewrite=True,
                        required_entities=[title],
                        resolved_references=[title],
                        standalone_query=f"请明确《{title}》与哪个对象、按什么指标比较。",
                    )
            elif intent == "meta":
                row.update(
                    query=f"第 {i + 1} 组讨论结束，请介绍你的论文问答流程以及引用格式。",
                    reference_answer="说明论文检索、证据反思与引用机制。",
                )
            else:
                out_count += 1
                row.update(
                    query=f"离开论文话题，请帮我写第 {out_count} 版生日聚会邀请函。",
                    expected_action="refuse",
                    should_refuse=True,
                    refusal_reason="out_of_scope",
                    reference_answer="仅处理科研论文相关任务。",
                )
            if not row["requires_rewrite"]:
                row["standalone_query"] = row["query"]
            result.append(row)
            # Reference history is only for isolated component tests. E2E uses actual outputs.
            history.extend(
                [
                    {"role": "user", "content": row["query"]},
                    {
                        "role": "assistant",
                        "content": f"讨论论文《{q['title']}》。"
                        + row["reference_answer"],
                    },
                ]
            )
    for row in [r for r in result if r["needs_retrieval"] and not r["should_refuse"]][
        :24
    ]:
        row["requires_second_retrieval"] = True
        row["retry_protocol"] = "controlled_first_pass_gold_removal"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    load_local_env()
    papers = read_jsonl(Path("corpus/papers_meta.jsonl"))
    chunks = read_jsonl(Path("corpus/chunks.jsonl"))
    grouped = defaultdict(list)
    for c in chunks:
        grouped[c["paper_id"]].append(c)
    output = Path("eval/question_bank")
    output.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        questions = list(
            pool.map(lambda p: question_bank(p, grouped[p["paper_id"]], output), papers)
        )
    bank = {p["paper_id"]: q for p, q in zip(papers, questions)}
    main_rows = build_dialogues(bank)
    retrieval = [
        {
            "id": f"R-{i + 1:03}",
            "query": q["query"],
            "ground_truth_chunk_ids": q["ground_truth_chunk_ids"],
            "support_quote": q["support_quote"],
            "reference_answer": q["answer"],
            "synthetic": True,
            "provenance": PROVENANCE,
            "annotation_status": q["annotation_status"],
        }
        for i, q in enumerate(q for group in questions for q in group)
    ]
    for name, rows in {
        "M": main_rows,
        "I": build_intent_dataset(main_rows),
        "R": retrieval,
        "B": build_boundary_dataset(main_rows),
        "E": build_e2e_dataset(main_rows),
    }.items():
        write_jsonl(Path("eval/data") / f"{name}.jsonl", rows)
    print(
        json.dumps(
            validate_datasets(Path("eval/data"), Path("corpus/chunks.jsonl")),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
