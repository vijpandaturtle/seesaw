# Circuit Finding

Identifying which subgraph of components implements a behavior. induction_heads and IOI are worked examples, not tools, so they're intentionally excluded here.

## Not yet implemented
- **circuit_evaluation** — Methodology for scoring whether a discovered circuit is actually correct/complete.
- **entity_binding** — Techniques for how a model tracks which entity a given piece of information belongs to.
- **circuit_tracing** — Anthropic-style attribution-graph tracing of a full circuit end to end.
- **universality** — Testing whether the same circuit/algorithm recurs across different models.
- **multimodal_interpretability** — Circuit-finding techniques applied to vision/multimodal models — depends on the nnsight backend decision (Part 8).
- **copy_suppression** — The specific mechanism by which some heads suppress naive copying behavior.
