"""FastMCP server for Lens, assembled from routers.

Run standalone:
    python -m lens.mcp_server.src.server

Exposes logit_lens, attention_pattern, ablation, activation_patching, and
direct_logit_attribution as MCP tools; results://last-bundle as a resource;
and tool_selection_guide as a prompt.
"""

from fastmcp import FastMCP

from .routers.prompts import register_mcp_prompts
from .routers.resources import register_mcp_resources
from .routers.tools import register_mcp_tools


def create_mcp_server() -> FastMCP:
    """Build a configured Lens FastMCP server instance.

    Importable for in-memory transport (see mcp_client), or run directly
    for stdio/http transport.
    """
    mcp = FastMCP("lens")
    register_mcp_tools(mcp)
    register_mcp_resources(mcp)
    register_mcp_prompts(mcp)
    return mcp


mcp = create_mcp_server()


if __name__ == "__main__":
    mcp.run()
