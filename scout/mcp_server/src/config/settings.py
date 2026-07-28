import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
FIRECRAWL_API_KEY: str | None = os.getenv("FIRECRAWL_API_KEY")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCOUT_ROOT: Path = Path(__file__).resolve().parents[4]  # scout/
# SEESAW_ARTIFACTS_DIR redirects output to shared storage. Running on
# Modal, the container filesystem is discarded when the function ends,
# so artifacts have to land on a mounted volume to outlive the run.
_ARTIFACTS_BASE = os.getenv("SEESAW_ARTIFACTS_DIR")
OUTPUTS_DIR: Path = (
    Path(_ARTIFACTS_BASE) / "scout" if _ARTIFACTS_BASE else SCOUT_ROOT / "outputs"
)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
SCOUT_MODEL: str = "claude-opus-4-5"
SCOUT_MAX_TOKENS: int = 8_000
SCOUT_TEMPERATURE: float = 1.0

# ── Scraping ──────────────────────────────────────────────────────────────────
MAX_AGE_ONE_WEEK: int = 604_800_000   # milliseconds — Firecrawl cache TTL
MAX_RETRIES: int = 3
MAX_CHARS: int = 8_000                 # truncate scraped content to keep context window manageable
