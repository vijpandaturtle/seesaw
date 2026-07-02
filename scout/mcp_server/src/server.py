"""FastMCP server for Scout, assembled from routers.

Run standalone:
    python -m scout.mcp_server.src.server

Exposes search_arxiv, search_arxiv_web, search_web, scrape_url, and
save_research_plan as MCP tools; system://status as a resource; and
research_instructions as a prompt.
"""

from fastmcp import FastMCP

from .routers.prompts import register_mcp_prompts
from .routers.resources import register_mcp_resources
from .routers.tools import register_mcp_tools


def create_mcp_server() -> FastMCP:
    """Build a configured Scout FastMCP server instance.

    Importable for in-memory transport (see mcp_client), or run directly
    for stdio/http transport.
    """
    mcp = FastMCP("scout")
    register_mcp_tools(mcp)
    register_mcp_resources(mcp)
    register_mcp_prompts(mcp)
    return mcp


mcp = create_mcp_server()


if __name__ == "__main__":
    mcp.run()
