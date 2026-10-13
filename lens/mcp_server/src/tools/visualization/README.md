# Visualization

Reading intermediate representations without intervening on the model — descriptive, not causal.

## Already built
The real computation lives in `seesaw/interp/visualization/` (pure functions,
no agent framing); this folder's sibling files (`../logit_lens.py`,
`../attention_pattern.py`, `../direct_logit_attribution.py`) are the thin
wrappers that call into the library and adapt the result for Lens's agent loop.
- `logit_lens` (built)
- `attention_pattern` (built)
- `direct_logit_attribution` (built)

## Not yet implemented
New techniques here get built the same way: the real logic goes in
`seesaw/interp/visualization/<name>.py` (returns an `InterpResult`), and a
thin `run_<name>` wrapper goes in `lens/mcp_server/src/tools/<name>.py`
(adapts it into an `ExperimentResult` and gets registered as an MCP tool).

- **tuned_lens** — Per-layer trained probe replacing the logit lens's reused final unembedding matrix.
- **jacobian_lens** — Uses the local Jacobian of remaining layers to project an intermediate hidden state more faithfully than logit lens.
