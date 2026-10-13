# Circuit Finding

Identifying which subgraph of components implements a behavior. induction_heads and IOI are worked examples, not tools, so they're intentionally excluded here.

## Built in `seesaw.interp.circuit_finding`
The computation lives in the library; the Lens wrapper is in `lens/mcp_server/src/tools/` and the tool is in `shared/lens_tools.py`.
- `circuit_evaluation` — faithfulness (only the circuit kept) and necessity (circuit removed), each against random circuits of the same size
- `edge_attribution_patching` — EAP / EAP-IG: every sender → receiver edge's estimated patching effect
- `path_patching` — patch single edges (a sender's output into one receiver's input) and measure
- `circuit_tracing` — attribution graphs from Neuronpedia's circuit-tracer (gemma-2-2b); generating one publishes it, so Lens does that only with SEESAW_NEURONPEDIA_PUBLISH=1

## Not yet implemented
- **entity_binding** — Techniques for how a model tracks which entity a given piece of information belongs to.
- **universality** — Testing whether the same circuit/algorithm recurs across different models.
- **multimodal_interpretability** — Circuit-finding techniques applied to vision/multimodal models — depends on the nnsight backend decision (Part 8).
- **copy_suppression** — The specific mechanism by which some heads suppress naive copying behavior.
