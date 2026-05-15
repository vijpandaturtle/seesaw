from firecrawl import FirecrawlApp
from langchain_core.tools import tool

from ..config import FIRECRAWL_API_KEY


@tool
def search_web(query: str, limit: int = 5) -> str:
    """Search the web for recent blog posts, GitHub repos, Colab notebooks,
    and community resources on AI safety and mechanistic interpretability.
    Returns URLs and content snippets. Use scrape_url to get full content
    from the most relevant results.

    Args:
        query: The search query
        limit: Number of results to return (default: 5)

    Returns:
        Search results with titles, URLs, and content snippets
    """
    if not FIRECRAWL_API_KEY:
        return "search_web unavailable: FIRECRAWL_API_KEY not set."

    app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

    try:
        response = app.search(query, limit=limit)

        raw_results = getattr(response, "data", None) or response
        if isinstance(raw_results, dict):
            raw_results = raw_results.get("data", [])

        if not raw_results:
            return f"No web results found for: '{query}'"

        results = []
        for r in raw_results:
            if isinstance(r, dict):
                title   = r.get("title", "N/A")
                url     = r.get("url", "N/A")
                snippet = r.get("description") or r.get("markdown", "")[:400]
            else:
                title   = getattr(r, "title", "N/A")
                url     = getattr(r, "url", "N/A")
                snippet = (getattr(r, "description", None) or getattr(r, "markdown", "") or "")[:400]

            results.append(
                f"**Title**: {title}\n"
                f"**URL**: {url}\n"
                f"**Snippet**: {snippet}...\n"
            )

        return f"Web results for '{query}':\n\n" + "\n---\n".join(results)

    except Exception as e:
        return f"search_web failed for '{query}': {str(e)}"
