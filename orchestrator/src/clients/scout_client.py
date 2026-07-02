"""Client for the Scout research-planning agent."""

from pathlib import Path

from scout.mcp_server.src.app.agent import build_agent
from scout.mcp_server.src.config.settings import OUTPUTS_DIR


def run_scout(
    research_question: str,
    plan_filename: str = "research_plan.md",
    thread_id: str = "scout-1",
    verbose: bool = True,
) -> Path:
    """Run the Scout agent on a research question.

    Scout searches arXiv and the web, then calls save_research_plan
    to write the plan to disk. This function streams the agent until
    that tool call completes, then returns the plan path.

    Args:
        research_question: The mech interp question to research.
        plan_filename: Filename for the saved plan (under scout/outputs/).
        thread_id: LangGraph thread ID for checkpointing.
        verbose: Print agent messages as they stream.

    Returns:
        Path to the saved research plan markdown file.

    Raises:
        RuntimeError: If Scout finishes without saving a plan.
    """
    agent  = build_agent()
    config = {"configurable": {"thread_id": thread_id}}

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

    plan_path = OUTPUTS_DIR / plan_filename
    if not plan_path.exists():
        raise RuntimeError(
            f"Scout completed but no plan found at {plan_path}. "
            "Check that save_research_plan was called."
        )

    print(f"✅ Scout done — plan saved to {plan_path}")
    return plan_path
