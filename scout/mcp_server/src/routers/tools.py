"""MCP tool registration for Scout.

Scout's tools are defined once as LangChain @tool functions (see ../tools/)
so the internal ReAct agent (app/agent.py) can use them directly. This
router re-exposes the same underlying logic as MCP tools via .invoke(),
so an external MCP client can call them individually without going
through the agent loop.
"""

from fastmcp import FastMCP

from .. import tools as scout_tools


def register_mcp_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def search_arxiv(query: str, max_results: int = 8) -> str:
        """Search arXiv for academic papers on mechanistic interpretability and AI safety.

        Args:
            query: The search query (e.g. 'GPT-2 indirect object identification circuit').
            max_results: Number of papers to return.

        Returns:
            Formatted list of papers with titles, authors, abstracts, and URLs.
        """
        return scout_tools.search_arxiv.invoke({"query": query, "max_results": max_results})

    @mcp.tool()
    def search_arxiv_web(query: str, limit: int = 5) -> str:
        """Search arxiv.org via web ranking — surfaces recent preprints the
        official API's relevance ranking can bury.

        Args:
            query: The search query.
            limit: Number of results to return.

        Returns:
            arXiv results with titles, URLs, and content snippets.
        """
        return scout_tools.search_arxiv_web.invoke({"query": query, "limit": limit})

    @mcp.tool()
    def search_web(query: str, limit: int = 5) -> str:
        """Search the web for blog posts, GitHub repos, Colab notebooks, and
        other community resources on mech interp / AI safety.

        Args:
            query: The search query.
            limit: Number of results to return.

        Returns:
            Search results with titles, URLs, and content snippets.
        """
        return scout_tools.search_web.invoke({"query": query, "limit": limit})

    @mcp.tool()
    def scrape_url(url: str) -> str:
        """Scrape and extract the full cleaned markdown content from a URL.

        Args:
            url: The URL to scrape.

        Returns:
            Cleaned markdown content, truncated to keep context manageable.
        """
        return scout_tools.scrape_url.invoke({"url": url})

    @mcp.tool()
    def save_research_plan(plan: str, filename: str = "research_plan.md") -> str:
        """Save the final structured Research Plan to disk. Signals the end
        of the Scout workflow — this is what Lens reads to run experiments.

        Args:
            plan: The complete research plan in markdown format.
            filename: Output filename.

        Returns:
            Confirmation message with the saved file path.
        """
        return scout_tools.save_research_plan.invoke({"plan": plan, "filename": filename})
