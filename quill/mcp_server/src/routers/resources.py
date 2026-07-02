from fastmcp import FastMCP

from ..resources.get_status_resource import register_status_resource


def register_mcp_resources(mcp: FastMCP) -> None:
    register_status_resource(mcp)
