import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
LENS_CLIENT_MODEL: str = "claude-sonnet-4-5"
LENS_CLIENT_TEMPERATURE: float = 0.2
