# Model Editing

Directly modifying model weights to change a specific stored fact or capability, rather than intervening at inference time.

## Built in `seesaw.interp.model_editing`
The computation lives in the library; the Lens wrapper is in `lens/mcp_server/src/tools/` and the tool is in `shared/lens_tools.py`.
- `directional_ablation` — project a direction out of the residual stream at every layer, against random directions and with a loss check (inference-time form of LEACE-style erasure)

## Not yet implemented
- **localized_fact_editing** — ROME/MEMIT-style targeted weight edits to change one stored fact.
- **leace** — Linear concept erasure — provably remove a linear concept from a representation.
- **unlearning** — Removing memorized training data or a specific capability from a trained model.
