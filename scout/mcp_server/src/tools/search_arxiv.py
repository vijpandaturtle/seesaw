import arxiv
from langchain_core.tools import tool

# One client for the process, not one per call. arxiv.Client enforces its
# politeness delay *per instance*, so constructing a fresh one each time resets
# the rate limiter — and Scout's ReAct loop issues several searches in a row,
# which is how a run ends up with HTTP 429 from export.arxiv.org.
_CLIENT = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)


@tool
def search_arxiv(query: str, max_results: int = 8) -> str:
    """Search arXiv for academic papers on mechanistic interpretability and AI safety.
    Use multiple targeted queries to cover different aspects of the research question.
    Best for finding foundational papers, circuit analysis, and formal methods.

    Args:
        query: The search query (e.g. 'GPT-2 indirect object identification circuit')
        max_results: Number of papers to return (default: 8)

    Returns:
        Formatted list of papers with titles, authors, abstracts, and URLs
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    try:
        for paper in _CLIENT.results(search):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            results.append(
                f"**Title**: {paper.title}\n"
                f"**Authors**: {authors}\n"
                f"**Published**: {paper.published.strftime('%Y-%m-%d')}\n"
                f"**Abstract**: {paper.summary[:600]}...\n"
                f"**URL**: {paper.entry_id}\n"
            )
    except Exception as exc:                # noqa: BLE001 — returned to the agent
        # A failed search is one tool call going wrong, not a reason to end the
        # research stage. Handing the agent the error lets it rephrase, try a
        # different query, or fall back to search_arxiv_web.
        partial = (
            f"\n\nPartial results before the failure:\n\n" + "\n---\n".join(results)
            if results
            else ""
        )
        return (
            f"search_arxiv failed for '{query}': {type(exc).__name__}: {exc}. "
            f"arXiv rate-limits repeated queries — wait before retrying, vary "
            f"the query, or use search_arxiv_web instead.{partial}"
        )

    if not results:
        return f"No papers found for query: '{query}'"

    return f"Found {len(results)} papers for '{query}':\n\n" + "\n---\n".join(results)
