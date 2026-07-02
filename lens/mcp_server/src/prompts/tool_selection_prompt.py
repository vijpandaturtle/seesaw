"""tool_selection_guide — how to choose a Lens tool for a given research question."""

from fastmcp import FastMCP

from ..tools import TOOL_REGISTRY


def register_tool_selection_prompt(mcp: FastMCP) -> None:
    @mcp.prompt()
    def tool_selection_guide() -> str:
        """Guidance for picking which Lens tool to run first.

        Lets a generic MCP client (not just the orchestrated workflow)
        decide which of the individually-callable tools to use.
        """
        return (
            "Available Lens tools: " + ", ".join(TOOL_REGISTRY.keys()) + "\n\n"
            "Order of operations for circuit_analysis tasks:\n"
            "  attention_pattern -> activation_patching -> ablation -> direct_logit_attribution\n\n"
            "attention_pattern and logit_lens are correlational — use them first to form a "
            "hypothesis about which layers/heads matter. activation_patching, ablation, and "
            "direct_logit_attribution are causal — use them to confirm the hypothesis. "
            "Never treat attention weight alone as evidence of causal importance."
        )
