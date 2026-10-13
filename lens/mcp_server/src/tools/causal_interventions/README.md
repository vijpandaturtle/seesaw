# Causal Interventions

Intervening on activations to test whether a component is actually load-bearing.

## Already built
The real computation lives in `seesaw/interp/causal_interventions/` (pure
functions, no agent framing); this folder's sibling files (`../ablation.py`,
`../activation_patching.py`) are the thin wrappers that call into the library
and adapt the result for Lens's agent loop.
- `ablation` (built)
- `activation_patching` (built)

## Not yet implemented
New techniques here get built the same way: the real logic goes in
`seesaw/interp/causal_interventions/<name>.py` (returns an `InterpResult`),
and a thin `run_<name>` wrapper goes in `lens/mcp_server/src/tools/<name>.py`
(adapts it into an `ExperimentResult` and gets registered as an MCP tool).

- **attribution_path_patching** — Approximates activation patching in one backward pass — the tool that makes automated circuit search (ACDC-style) tractable.
- **causal_mediators** — Systematic selection of which components to treat as candidate causal mediators before patching.
- **refined_attribution** — Higher-fidelity attribution methods beyond basic attribution patching.
- **self_repair** — Detects when ablating one component causes others to compensate, masking the true effect.
- **causal_abstraction** — Formal framework for testing whether a high-level causal variable is implemented by a low-level circuit.
