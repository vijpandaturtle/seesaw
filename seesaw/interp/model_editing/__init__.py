"""Model Editing — Directly modifying model weights to change a specific stored fact or capability, rather than intervening at inference time.

Nothing implemented yet. See `lens/mcp_server/src/tools/model_editing/README.md` for the planned techniques and
their one-line descriptions.

Pattern to follow when implementing one (matches visualization/ and
causal_interventions/, the two categories already built):

    def compute(model, prompts, output_dir, *technique_specific_args, **kwargs) -> InterpResult:
        ...
        return InterpResult(data={...}, plot_paths=[...])

Pure function: no agent framing, no MCP, no ExperimentResult. Lens wraps it
afterward in a thin `run_<name>` function under `lens/mcp_server/src/tools/`.
"""
