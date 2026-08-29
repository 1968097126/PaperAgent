"""组件级细粒度评测：意图分类、检索、拒答、引用、端到端、记忆一致性。

同样是包标志文件，让 eval 目录可以被 `from eval.metrics import ...` 这样导入。

后续会包含（对应工作规划 Phase 6）：
    dataset.jsonl   100~150 条人工构造的多轮对话评测样本，每条标注：
                      - 所属意图（五分类之一）
                      - 指代消解后的目标（该轮 query 里的代词应该指向哪个实体/上一轮的哪个结论）
                      - 检索目标（应该命中的 chunk_id / 论文），用来算 Recall@K
                      - 拒答标签（这条问题是否应该被 Guardrails/Reflect 拦下，不强行作答）
    metrics.py       各项指标的计算函数：Intent Accuracy、Query改写准确率、Retrieval Recall@K、
                      拒答准确率、引用准确率、Pipeline Success Rate、Memory一致性
    run_eval.py       跑一遍 dataset.jsonl，调用 pipeline.orchestrator 逐条执行，
                      再用 metrics.py 汇总出评测报告（对应简历"项目成果"里的各项百分比）

Phase 0 阶段这里先只放这份说明性的 __init__.py。
"""
