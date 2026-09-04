"""Research-agent pipeline package.

The package combines the Phase 3 harness with the Phase 4 research pipeline:

* ``agent.AgentState`` keeps semantic, retrieval, citation, memory, and trace state.
* ``agent.run_research`` exposes only ``search_papers`` and ``retrieve_chunk``
  schemas and enforces the same allowlist immediately before tool dispatch.
* ``pipeline.trace`` provides one TurnTrace/StageTrace implementation shared by
  the inherited agent loop and the explicit seven-stage pipeline.
* ``pipeline.orchestrator.ResearchPipeline`` runs Guardrails → NLU → Retrieve
  → Rerank → Reflect → Generate → Memory, records bounded degradation, and
  commits one coherent state update per user turn.
* ``pipeline.client.NativeProviderClient`` adapts the inherited streaming
  Provider registry without exposing tools to individual semantic stages.
"""
