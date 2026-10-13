"""Circuit Finding — Identifying which subgraph of components implements a behavior. induction_heads and IOI are worked examples, not tools, so they're intentionally excluded here.

Built:
- circuit_evaluation — faithfulness and necessity of a head set, against random circuits
- edge_attribution_patching — EAP / EAP-IG score for every sender → receiver edge (nominates)
- path_patching — patch single edges for real (verifies)
- circuit_tracing — attribution graphs from Neuronpedia (circuit-tracer on gemma-2-2b)

Planned: see `lens/mcp_server/src/tools/circuit_finding/README.md`.
"""
