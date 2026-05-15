from firecrawl import FirecrawlApp
from langchain_core.tools import tool

from ..config import FIRECRAWL_API_KEY


@tool
def search_arxiv_web(query: str, limit: int = 5) -> str:
    """Search arxiv.org via Firecrawl for papers on mechanistic interpretability and AI safety.
    Complements search_arxiv: while search_arxiv uses the official API (structured, keyword-matched),
    this tool uses web search ranking — better for surfacing recent preprints, finding papers
    that cite a key work, or when the API's relevance ranking buries what you need.

    Args:
        query: The search query (e.g. 'transformer circuits attention head superposition')
        limit: Number of results to return (default: 5)

    Returns:
        Search results from arxiv.org with titles, URLs, and content snippets
    """
    if not FIRECRAWL_API_KEY:
        return "search_arxiv_web unavailable: FIRECRAWL_API_KEY not set."

    app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

    # Scope the search to arxiv.org
    arxiv_query = f"site:arxiv.org {query}"

    try:
        response = app.search(arxiv_query, limit=limit)

        raw_results = getattr(response, "data", None) or response
        if isinstance(raw_results, dict):
            raw_results = raw_results.get("data", [])

        if not raw_results:
            return f"No arxiv.org results found for: '{query}'"

        results = []
        for r in raw_results:
            if isinstance(r, dict):
                title   = r.get("title", "N/A")
                url     = r.get("url", "N/A")
                snippet = r.get("description") or r.get("markdown", "")
            else:
                title   = getattr(r, "title", "N/A")
                url     = getattr(r, "url", "N/A")
                snippet = getattr(r, "description", None) or getattr(r, "markdown", "") or ""

            # Only keep actual arxiv.org links
            if "arxiv.org" not in str(url):
                continue

            results.append(
                f"**Title**: {title}\n"
                f"**URL**: {url}\n"
                f"**Snippet**: {str(snippet)[:400]}...\n"
            )

        if not results:
            return f"No arxiv.org results found for: '{query}'"

        return f"arxiv.org web results for '{query}':\n\n" + "\n---\n".join(results)

    except Exception as e:
        return f"search_arxiv_web failed for '{query}': {str(e)}"
