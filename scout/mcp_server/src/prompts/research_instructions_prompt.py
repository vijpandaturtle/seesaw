"""research_instructions — the Scout system prompt, retrievable by MCP clients."""

from fastmcp import FastMCP

from ..config import SCOUT_SYSTEM_PROMPT


def register_research_instructions_prompt(mcp: FastMCP) -> None:
    @mcp.prompt()
    def research_instructions() -> str:
        """The full Scout workflow: decompose → search arXiv → search web →
        scrape → synthesise → save. Lets a generic MCP client (not just the
        built-in ReAct agent) drive Scout's tools with the same instructions.
        """
        return SCOUT_SYSTEM_PROMPT
