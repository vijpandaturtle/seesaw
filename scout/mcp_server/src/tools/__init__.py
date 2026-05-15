from .search_arxiv import search_arxiv
from .search_arxiv_web import search_arxiv_web
from .search_web import search_web
from .scrape_url import scrape_url
from .save_research_plan import save_research_plan

SCOUT_TOOLS = [
    search_arxiv,
    search_arxiv_web,
    search_web,
    scrape_url,
    save_research_plan,
]

__all__ = [
    "search_arxiv",
    "search_arxiv_web",
    "search_web",
    "scrape_url",
    "save_research_plan",
    "SCOUT_TOOLS",
]
