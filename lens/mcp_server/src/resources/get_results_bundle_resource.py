"""results://last-bundle — the most recently saved Lens results bundle."""

import json

from fastmcp import FastMCP

from ..config import OUTPUTS_DIR


def register_results_bundle_resource(mcp: FastMCP) -> None:
    @mcp.resource("results://last-bundle")
    def get_last_results_bundle() -> str:
        """Return the most recently saved Lens results bundle from disk.

        The results bundle is produced by the orchestrated research-plan
        workflow (not by calling the individual tools), and is what gets
        handed to Quill for critique.
        """
        bundle_path = OUTPUTS_DIR / "results_bundle.json"
        if not bundle_path.exists():
            return json.dumps({"error": f"No results bundle found at {bundle_path}"})
        return bundle_path.read_text()
