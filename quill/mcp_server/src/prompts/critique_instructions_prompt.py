"""critique_instructions — Quill's three node-level system prompts,
retrievable by MCP clients that want to drive the critique logic themselves
rather than going through the LangGraph workflow.
"""

from fastmcp import FastMCP

from ..config import (
    CRITIQUE_EXPERIMENTS_SYSTEM,
    GENERATE_FOLLOWUPS_SYSTEM,
    IDENTIFY_GAPS_SYSTEM,
)


def register_critique_instructions_prompt(mcp: FastMCP) -> None:
    @mcp.prompt()
    def critique_instructions() -> str:
        """The full Quill critique workflow, concatenated: assess each
        experiment for validity → identify missing experiments →
        generate concrete follow-up specs for Lens.
        """
        return "\n\n---\n\n".join([
            CRITIQUE_EXPERIMENTS_SYSTEM,
            IDENTIFY_GAPS_SYSTEM,
            GENERATE_FOLLOWUPS_SYSTEM,
        ])
