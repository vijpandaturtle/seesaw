"""FastMCP server for Quill, assembled from routers.

Run standalone:
    python -m quill.mcp_server.src.server

Exposes critique_experiment_bundle as an MCP tool; system://status as a
resource; and critique_instructions as a prompt.
"""

from fastmcp import FastMCP

from .routers.prompts import register_mcp_prompts
from .routers.resources import register_mcp_resources
from .routers.tools import register_mcp_tools


def create_mcp_server() -> FastMCP:
    """Build a configured Quill FastMCP server instance.

    Importable for in-memory transport (see mcp_client), or run directly
    for stdio/http transport.
    """
    mcp = FastMCP("quill")
    register_mcp_tools(mcp)
    register_mcp_resources(mcp)
    register_mcp_prompts(mcp)
    return mcp


mcp = create_mcp_server()


if __name__ == "__main__":
    mcp.run()
