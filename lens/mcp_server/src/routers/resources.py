from fastmcp import FastMCP

from ..resources.get_results_bundle_resource import register_results_bundle_resource


def register_mcp_resources(mcp: FastMCP) -> None:
    register_results_bundle_resource(mcp)
