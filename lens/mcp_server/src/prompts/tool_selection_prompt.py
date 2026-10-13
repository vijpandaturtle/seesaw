"""tool_selection_guide — how to choose a Lens tool for a given research question."""

from fastmcp import FastMCP

from shared.lens_tools import markdown_table


def register_tool_selection_prompt(mcp: FastMCP) -> None:
    @mcp.prompt()
    def tool_selection_guide() -> str:
        """Guidance for picking which Lens tool to run first.

        Lets a generic MCP client (not just the orchestrated workflow)
        decide which of the individually-callable tools to use.
        """
        return (
            "Available Lens tools:\n\n" + markdown_table() + "\n\n"
            "Nominate with a cheap tool, then verify with a causal one:\n"
            "- Circuits: direct_logit_attribution + attribution_patching -> head_patching "
            "(or ablation with `heads`) -> circuit_evaluation\n"
            "- Is an attribute used (bias audits): counterfactual -> linear_probe -> "
            "directional_ablation and steering_vector\n"
            "- Which features: feature_search / feature_attribution -> feature_clamping\n\n"
            "Descriptive tools show association and estimates only rank candidates; neither is "
            "evidence that a component is necessary. Never treat attention weight alone as "
            "evidence of causal importance."
        )
