import time

from firecrawl import FirecrawlApp
from langchain_core.tools import tool

from ..config import FIRECRAWL_API_KEY, MAX_RETRIES, MAX_CHARS


@tool
def scrape_url(url: str) -> str:
    """Scrape and extract the full cleaned markdown content from a URL using Firecrawl.
    Use this after search_web to get the complete content of a specific page —
    full paper text, entire blog post, complete README, etc.

    Args:
        url: The URL to scrape

    Returns:
        Cleaned markdown content from the page (truncated at 8000 chars)
    """
    if not FIRECRAWL_API_KEY:
        return "scrape_url unavailable: FIRECRAWL_API_KEY not set."

    app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

    for attempt in range(MAX_RETRIES):
        try:
            res = app.scrape_url(url, formats=["markdown"])

            if isinstance(res, dict):
                title    = res.get("metadata", {}).get("title", "N/A")
                markdown = res.get("markdown", "")
            else:
                title    = getattr(getattr(res, "metadata", None), "title", "N/A") or "N/A"
                markdown = getattr(res, "markdown", "") or ""

            if not markdown.strip():
                return f"No content returned for {url}"

            if len(markdown) > MAX_CHARS:
                markdown = markdown[:MAX_CHARS] + "\n\n[... content truncated ...]"

            return f"# {title}\nSource: {url}\n\n{markdown}"

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = 5 * (2 ** attempt)   # exponential backoff: 5s, 10s
                print(f"  ⚠️ scrape attempt {attempt + 1} failed for {url} — retrying in {delay}s")
                time.sleep(delay)
            else:
                return f"Failed to scrape {url} after {MAX_RETRIES} attempts: {str(e)}"

    return f"Failed to scrape {url}"
