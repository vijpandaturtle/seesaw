"""Black Box — No model internals required at all — input/output only. Cheapest tool here, and the only one that survives a closed model or a backend change.

Nothing implemented yet. See `lens/mcp_server/src/tools/black_box/README.md` for the planned techniques and
their one-line descriptions.

Pattern to follow when implementing one (matches visualization/ and
causal_interventions/, the two categories already built):

    def compute(model, prompts, output_dir, *technique_specific_args, **kwargs) -> InterpResult:
        ...
        return InterpResult(data={...}, plot_paths=[...])

Pure function: no agent framing, no MCP, no ExperimentResult. Lens wraps it
afterward in a thin `run_<name>` function under `lens/mcp_server/src/tools/`.
"""
