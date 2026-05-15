import arxiv
from langchain_core.tools import tool


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
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    for paper in client.results(search):
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

    if not results:
        return f"No papers found for query: '{query}'"

    return f"Found {len(results)} papers for '{query}':\n\n" + "\n---\n".join(results)
