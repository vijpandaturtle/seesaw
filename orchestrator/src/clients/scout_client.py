"""Client for the Scout research-planning agent."""

import time
from pathlib import Path

from scout.mcp_server.src.app.agent import build_agent
from scout.mcp_server.src.config.settings import OUTPUTS_DIR


def run_scout(
    research_question: str,
    thread_id: str = "scout-1",
    verbose: bool = True,
) -> Path:
    """Run the Scout agent on a research question.

    Scout searches arXiv and the web, then calls save_research_plan
    to write the plan to disk. This function streams the agent until
    that tool call completes, then returns the plan path.

    Scout's LLM picks its own descriptive filename when saving (e.g.
    "gender_bias_attention_heads_research_plan.md", not always the
    literal "research_plan.md"), so instead of checking a fixed name,
    this looks for the newest "*research_plan*.md" file written to
    OUTPUTS_DIR during this run.

    Args:
        research_question: The mech interp question to research.
        thread_id: LangGraph thread ID for checkpointing.
        verbose: Print agent messages as they stream.

    Returns:
        Path to the saved research plan markdown file.

    Raises:
        RuntimeError: If Scout finishes without saving a plan.
    """
    agent      = build_agent()
    config     = {"configurable": {"thread_id": thread_id}}
    start_time = time.time()

    print(f"🔍 Scout starting — question: {research_question!r}")

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": research_question}]},
        config=config,
        stream_mode="values",
    ):
        if not verbose:
            continue
        last = chunk["messages"][-1]
        if hasattr(last, "content") and last.content:
            # Only print non-empty text content (skip tool call objects)
            if isinstance(last.content, str):
                print(last.content[:500])

    candidates = [
        p for p in OUTPUTS_DIR.glob("*research_plan*.md")
        if p.stat().st_mtime >= start_time
    ]
    if not candidates:
        raise RuntimeError(
            f"Scout completed but no *research_plan*.md file appeared in {OUTPUTS_DIR}. "
            "Check that save_research_plan was called."
        )

    plan_path = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"✅ Scout done — plan saved to {plan_path}")
    return plan_path
