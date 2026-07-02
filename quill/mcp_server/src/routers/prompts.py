from fastmcp import FastMCP

from ..prompts.critique_instructions_prompt import register_critique_instructions_prompt


def register_mcp_prompts(mcp: FastMCP) -> None:
    register_critique_instructions_prompt(mcp)
