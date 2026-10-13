# Causal Interventions

Intervening on activations to test whether a component is actually load-bearing.

## Already built
The real computation lives in `seesaw/interp/causal_interventions/` (pure
functions, no agent framing); this folder's sibling files (`../ablation.py`,
`../activation_patching.py`) are the thin wrappers that call into the library
and adapt the result for Lens's agent loop.
- `ablation` (built; zero / mean / resample, full sweep or a shortlist of heads)
- `activation_patching` (built)
- `attribution_patching` (built) — every head's and MLP's patching effect estimated from one backward pass; nominates
- `head_patching` (built) — real patching of a shortlist of heads, alone and together; verifies

## Not yet implemented
New techniques here get built the same way: the real logic goes in
`seesaw/interp/causal_interventions/<name>.py` (returns an `InterpResult`),
and a thin `run_<name>` wrapper goes in `lens/mcp_server/src/tools/<name>.py`
(adapts it into an `ExperimentResult`), registered in `TOOL_REGISTRY` and
catalogued in `shared/lens_tools.py` (tests/test_lens_tools.py checks the two agree).

- **causal_mediators** — Systematic selection of which components to treat as candidate causal mediators before patching.
- **refined_attribution** — Higher-fidelity attribution methods beyond basic attribution patching.
- **self_repair** — Detects when ablating one component causes others to compensate, masking the true effect.
- **causal_abstraction** — Formal framework for testing whether a high-level causal variable is implemented by a low-level circuit.
