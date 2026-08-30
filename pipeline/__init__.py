"""七阶段显式执行链路：Guardrails -> NLU -> 混合检索 -> 重排序 -> Reflect -> 生成 -> Memory。

同 retrieval/__init__.py 一样，这个文件的存在让 pipeline 目录成为一个可导入的包。

这个包里的模块不是"LLM 自己决定调不调用的工具"，而是每一轮对话都会按固定顺序依次
执行的编排步骤（这也是简历里"显式执行链路"这个说法的来源——顺序是写死在代码里的，
不是模型自由决定的）。之所以能这样做，是因为复用了 nano-claude-code 原有的工具执行内核
（tool_registry.py 的注册/分发/截断机制、agent.py 的状态管理），检索类的具体动作
（search_papers / retrieve_chunk / rerank）仍然注册成 ToolDef，只是改成由本包里的
orchestrator.py 按固定顺序主动调用，而不是等 LLM 通过 function-calling 来决定调用时机。

已完成（Phase 3）：
    trace.py        StageTrace/TurnTrace 数据结构：记录每一轮各阶段的输入/输出/耗时，
                    是"全链路可追溯设计"的具体实现；已接入 agent.py 的 run() 循环，
                    每次调用会往 AgentState.turn_traces 里追加一条记录
    （agent.py 同时加了 tool_names 参数——白名单之外的工具既不出现在发给LLM的
     schema里，真调用了也会在tool_registry执行之前就被拦下，为Phase 4的"混合检索"
     阶段只暴露search_papers/retrieve_chunk两个领域工具做好了准备）

待实现（Phase 4）：
    guardrails.py   前置判断问题是否落在语料库覆盖范围内；超范围直接返回澄清话术，
                    不进入下面的检索/生成，避免"证据库里根本没有还硬答"
    nlu.py          五分类（新检索/追问/澄清/元问题/越界）+ 结合历史做指代消解和
                    query 改写，输出下游检索能直接用的独立 query
    reflect.py      判断精排后的证据是否足够支撑回答；不够就触发二次检索，
                    或者直接告诉用户"当前语料库未覆盖"而不是编造答案
    generate.py     基于证据生成最终回答，并要求标注引用来源
    memory.py       多级记忆压缩：随对话轮次增长，把早期历史压缩为摘要，
                    在 compaction.py 现有两层压缩机制基础上扩展
    orchestrator.py 把以上几步按固定顺序串起来，驱动一轮对话的完整执行，
                    过程中往 TurnTrace 里记录每一步
"""
