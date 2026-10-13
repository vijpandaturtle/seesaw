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
    ├── visualization/         — descriptive: logit lens, attention pattern, DLA
    ├── causal_interventions/  — causal: ablation, activation patching
    └── ...                    — more categories added as they're built;
                                  see lens/mcp_server/src/tools/<category>/README.md
                                  for what's planned in each

Needs the `[experiments]` extra (torch, transformer-lens, matplotlib). Nothing
in this package is imported eagerly by plain `import seesaw` — see
seesaw/__init__.py's lazy attribute resolution.
"""
