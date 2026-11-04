"""Client for the Scout research-planning agent."""

import time
from pathlib import Path
from typing import Callable

from scout.mcp_server.src.app.agent import build_agent
from scout.mcp_server.src.config.settings import OUTPUTS_DIR
from shared.db import events

# How many times Scout is asked to fix a plan that fails `check`.
MAX_REVISIONS = 2

_REVISE = (
    "The plan you saved can't be used yet. Its pathway block failed validation:\n\n"
    "{errors}\n\n"
    "Fix every problem above and call save_research_plan again with the full "
    "corrected plan, including exactly one fenced ```json pathway block. "
    "Don't search again unless you need to."
)


def run_scout(
    research_question: str,
    thread_id: str = "scout-1",
    verbose: bool = True,
    output_dir: Path = OUTPUTS_DIR,
    brief: str | None = None,
    check: Callable[[Path], list[str]] | None = None,
    max_revisions: int = MAX_REVISIONS,
    search: bool = True,
) -> Path:
    """Run the Scout agent on a research question.

    Scout searches arXiv and the web (when `search` is on), then calls save_research_plan
    to write the plan to disk. This function streams the agent until
    that tool call completes, then returns the plan path.

    Scout's LLM picks its own descriptive filename when saving (e.g.
    "gender_bias_attention_heads_research_plan.md", not always the
    literal "research_plan.md"), so instead of checking a fixed name,
    this looks for the newest "*research_plan*.md" file written to
    output_dir during this run.

    With `check`, each saved plan is checked and, while it returns errors,
    Scout is sent them on the same thread (so it keeps its research) and
    asked to save a corrected plan, up to `max_revisions` times. The last
    plan is returned either way: the caller decides what errors mean.

    Args:
        research_question: The mech interp question to research.
        thread_id: LangGraph thread ID for checkpointing.
        verbose: Print agent messages as they stream.
        output_dir: Where the plan is saved; a job passes its own folder.
        brief: Extra instructions sent with the question, e.g. the pathway
            whose inputs Scout has to choose (catalog.scout_brief).
        check: Plan path -> list of problems; [] when the plan is usable.
        max_revisions: How many corrected plans to ask for.
        search: Give Scout its literature search tools.

    Returns:
        Path to the saved research plan markdown file.

    Raises:
        RuntimeError: If Scout finishes without saving a plan.
    """
    agent  = build_agent(output_dir=output_dir, search=search)
    config = {"configurable": {"thread_id": thread_id}, "callbacks": events.callbacks()}

    print(f"🔍 Scout starting — question: {research_question!r}"
          + ("" if search else " (literature search off)"))

    message = research_question if brief is None else f"{research_question}\n\n{brief}"
    plan_path = _turn(agent, config, message, output_dir, verbose)
    for attempt in range(1, max_revisions + 1):
        errors = check(plan_path) if check else []
        if check:
            events.emit(events.CHECK, "pathway inputs", status=events.FAILED if errors else events.OK,
                        summary=f"{len(errors)} problem(s); asking Scout to revise "
                                f"({attempt}/{max_revisions})" if errors else "inputs are valid",
                        data={"plan": plan_path.name, "errors": errors})
        if not errors:
            break
        print(f"⚠️  Scout plan failed its check (revision {attempt}/{max_revisions}):")
        for e in errors:
            print(f"   - {e}")
        plan_path = _turn(agent, config, _REVISE.format(errors="\n".join(f"- {e}" for e in errors)),
                          output_dir, verbose)

    print(f"✅ Scout done — plan saved to {plan_path}")
    return plan_path


def _turn(agent, config: dict, message: str, output_dir: Path, verbose: bool) -> Path:
    """Send one user message, stream the agent, return the plan it saved."""
    start_time = time.time()
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": message}]},
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
        p for p in output_dir.glob("*research_plan*.md")
        if p.stat().st_mtime >= start_time
    ]
    if not candidates:
        raise RuntimeError(
            f"Scout completed but no *research_plan*.md file appeared in {output_dir}. "
            "Check that save_research_plan was called."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)
