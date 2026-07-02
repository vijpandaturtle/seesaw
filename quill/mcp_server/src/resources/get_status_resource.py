"""system://status — Quill server configuration and past critique reports."""

import json

from fastmcp import FastMCP

from ..config import OUTPUTS_DIR, QUILL_MODEL, QUILL_TEMPERATURE


def register_status_resource(mcp: FastMCP) -> None:
    @mcp.resource("system://status")
    def get_status() -> str:
        """Report Quill's model config and which critique reports have
        already been saved to disk (from nodes/save_critique.py)."""
        status = {
            "model": QUILL_MODEL,
            "temperature": QUILL_TEMPERATURE,
            "outputs_dir": str(OUTPUTS_DIR),
            "saved_reports": sorted(p.name for p in OUTPUTS_DIR.glob("critique_*.md")),
        }
        return json.dumps(status, indent=2)
