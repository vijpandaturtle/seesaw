import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

# ── Paths ─────────────────────────────────────────────────────────────────────
QUILL_ROOT: Path = Path(__file__).resolve().parents[4]   # quill/
# SEESAW_ARTIFACTS_DIR redirects output to shared storage. Running on
# Modal, the container filesystem is discarded when the function ends,
# so artifacts have to land on a mounted volume to outlive the run.
_ARTIFACTS_BASE = os.getenv("SEESAW_ARTIFACTS_DIR")
OUTPUTS_DIR: Path = (
    Path(_ARTIFACTS_BASE) / "quill" if _ARTIFACTS_BASE else QUILL_ROOT / "outputs"
)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
# Opus for deep analytical critique — Quill is the most reasoning-intensive agent
QUILL_MODEL: str = "claude-opus-4-5"
QUILL_TEMPERATURE: float = 0.2
