"""Research-agent pipeline package.

Phase 3 establishes the harness boundary used by later orchestration code:

* ``agent.AgentState`` keeps semantic, retrieval, citation, memory, and trace state.
* ``agent.run_research`` exposes only ``search_papers`` and ``retrieve_chunk``
  schemas and enforces the same allowlist immediately before tool dispatch.
* ``pipeline.trace`` provides one TurnTrace/StageTrace implementation shared by
  the inherited agent loop and the explicit seven-stage pipeline.

The Guardrails → NLU → Retrieve → Rerank → Reflect → Generate → Memory
orchestrator belongs to Phase 4 and is intentionally kept separate from this
package-level compatibility contract.
"""
