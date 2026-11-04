# Model Editing

Directly modifying model weights to change a specific stored fact or capability, rather than intervening at inference time.

## Built in `seesaw.interp.model_editing`
The computation lives in the library; the Lens wrapper is in `lens/mcp_server/src/tools/` and the tool is in `shared/lens_tools.py`.
- `directional_ablation` — project a direction out of the residual stream at every layer, against random directions and with a loss check; `orthogonalized` bakes the same removal into the weights
- `rome` — rank-one edit to one MLP's output weights to rewrite a fact, scored on efficacy, paraphrase and specificity (library and pathways only; no Lens tool yet)
- `leace` — linear concept erasure (Belrose et al. 2023), one eraser per residual-stream site (library and pathways only)

## Not yet implemented
- **memit** — ROME's multi-layer, many-facts-at-once successor.
- **unlearning** — Removing memorized training data or a specific capability from a trained model.
