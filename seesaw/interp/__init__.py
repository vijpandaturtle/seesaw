"""Seesaw's mechanistic interpretability toolkit — the actual science.

This package has no knowledge of agents, MCP, or LangGraph. Every function
here takes a loaded model plus plain arguments and returns an `InterpResult`
(raw numbers and, where relevant, saved plot paths) — nothing agent-shaped.

`lens/mcp_server/src/tools/` is the thin layer on top: it calls into here and
wraps the result into an `ExperimentResult` (adds a `summary` slot for the
LLM's interpretation, a `status`, and registers as an MCP tool). That means
every function in this package is independently usable — from a notebook,
from an eval grader, or from a future agent that isn't Lens — without pulling
in any of that.

Organized by category, following the taxonomy at learnmechinterp.com:

    interp/
    ├── metrics.py             — logit diff and recovery, shared by every tool
    ├── visualization/         — descriptive: logit lens, attention pattern, DLA
    ├── causal_interventions/  — ablation, activation / attribution / head patching
    ├── circuit_finding/       — circuit evaluation (faithfulness vs random circuits)
    ├── black_box/             — counterfactual evaluation (inputs and outputs only)
    ├── probing/               — linear probes and difference-of-means directions
    ├── steering/              — steering vectors, SAE feature clamping
    ├── model_editing/         — directional ablation
    └── feature_extraction/    — SAE loading, feature attribution, Neuronpedia client
    (feature_geometry, hidden_state_decoding, model_diffing, weight_space:
     planned; see lens/mcp_server/src/tools/<category>/README.md)

Gradient-based tools (attribution_patching, feature_attribution) are
estimates that nominate candidates; the intervention tools verify them.

Needs the `[experiments]` extra (torch, transformer-lens, matplotlib). Nothing
in this package is imported eagerly by plain `import seesaw` — see
seesaw/__init__.py's lazy attribute resolution.
"""
