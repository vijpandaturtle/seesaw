from fastmcp import FastMCP

from ..prompts.tool_selection_prompt import register_tool_selection_prompt


def register_mcp_prompts(mcp: FastMCP) -> None:
    register_tool_selection_prompt(mcp)
