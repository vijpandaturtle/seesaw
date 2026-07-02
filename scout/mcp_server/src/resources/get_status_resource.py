"""system://status — Scout server configuration and readiness."""

import json

from fastmcp import FastMCP

from ..config import FIRECRAWL_API_KEY, OUTPUTS_DIR, SCOUT_MAX_TOKENS, SCOUT_MODEL, SCOUT_TEMPERATURE


def register_status_resource(mcp: FastMCP) -> None:
    @mcp.resource("system://status")
    def get_status() -> str:
        """Report Scout's model config and which optional tools are available.

        search_web and scrape_url require FIRECRAWL_API_KEY — this resource
        lets a client check whether those tools will actually work before
        calling them.
        """
        status = {
            "model": SCOUT_MODEL,
            "max_tokens": SCOUT_MAX_TOKENS,
            "temperature": SCOUT_TEMPERATURE,
            "outputs_dir": str(OUTPUTS_DIR),
            "firecrawl_configured": bool(FIRECRAWL_API_KEY),
            "saved_plans": sorted(p.name for p in OUTPUTS_DIR.glob("*.md")),
        }
        return json.dumps(status, indent=2)
