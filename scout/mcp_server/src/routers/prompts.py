from fastmcp import FastMCP

from ..prompts.research_instructions_prompt import register_research_instructions_prompt


def register_mcp_prompts(mcp: FastMCP) -> None:
    register_research_instructions_prompt(mcp)
