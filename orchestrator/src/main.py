"""CLI entry point for Seesaw.

Usage:
    python -m orchestrator.src.main --question "What heads mediate IOI in GPT-2 Small?"
    python -m orchestrator.src.main --question "..." --skip-hitl
"""

import argparse
import sys

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="seesaw",
        description="Seesaw — agentic mechanistic interpretability pipeline",
    )
    parser.add_argument(
        "--question", "-q",
        required=True,
        help="Research question for Scout to investigate",
    )
    parser.add_argument(
        "--skip-hitl",
        action="store_true",
        default=False,
        help="Skip human-in-the-loop checkpoints (automated runs)",
    )

    args = parser.parse_args()

    result = run_pipeline(
        research_question=args.question,
        skip_hitl=args.skip_hitl,
    )

    if result["report"] is None:
        print("\nPipeline stopped early.")
        sys.exit(1)


if __name__ == "__main__":
    main()
