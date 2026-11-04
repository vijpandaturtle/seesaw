"""Model Editing — Directly modifying model weights to change a specific stored fact or capability, rather than intervening at inference time.

Built: directional_ablation (project a direction out of the residual stream everywhere, or
bake that into the weights), rome (rank-one fact edits), leace (linear concept erasure).
Planned: see `lens/mcp_server/src/tools/model_editing/README.md`.
"""
