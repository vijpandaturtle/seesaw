import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

# ── Paths ─────────────────────────────────────────────────────────────────────
QUILL_ROOT: Path = Path(__file__).resolve().parents[4]   # quill/
OUTPUTS_DIR: Path = QUILL_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
# Opus for deep analytical critique — Quill is the most reasoning-intensive agent
QUILL_MODEL: str = "claude-opus-4-5"
QUILL_TEMPERATURE: float = 0.2
