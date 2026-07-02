"""LangGraph workflow for the Lens agent.

Graph structure:
    parse_plan → load_model → run_experiment → interpret_result → check_followup
                                    ▲                                      │
                                    └──────── more experiments? ───────────┘
                                                                           │
                                                                   collect_results → END
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from ..nodes import (
    check_followup,
    collect_results,
    interpret_result,
    load_model_node,
    parse_plan,
    run_experiment,
)


# ── LangGraph state ────────────────────────────────────────────────────────────
class LensState(TypedDict):
    research_plan:     str
    research_question: str
    experiment_queue:  list[dict]
    model_name:        str
    last_result:       dict | None
    followup_count:    int
    results:           list[dict]
    bundle:            dict | None


# ── Router ────────────────────────────────────────────────────────────────────
def route_after_followup(state: LensState) -> str:
    return "run_experiment" if state.get("experiment_queue") else "collect_results"


# ── Graph builder ─────────────────────────────────────────────────────────────
def build_lens_graph(checkpointer=None):
    """Build and compile the Lens LangGraph workflow.

    Args:
        checkpointer: LangGraph checkpointer. Defaults to MemorySaver.

    Returns:
        Compiled StateGraph ready to invoke or stream.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    workflow = StateGraph(LensState)
    workflow.add_node("parse_plan",       parse_plan)
    workflow.add_node("load_model",       load_model_node)
    workflow.add_node("run_experiment",   run_experiment)
    workflow.add_node("interpret_result", interpret_result)
    workflow.add_node("check_followup",   check_followup)
    workflow.add_node("collect_results",  collect_results)

    workflow.set_entry_point("parse_plan")
    workflow.add_edge("parse_plan",       "load_model")
    workflow.add_edge("load_model",       "run_experiment")
    workflow.add_edge("run_experiment",   "interpret_result")
    workflow.add_edge("interpret_result", "check_followup")
    workflow.add_conditional_edges(
        "check_followup",
        route_after_followup,
        {"run_experiment": "run_experiment", "collect_results": "collect_results"},
    )
    workflow.add_edge("collect_results", END)

    return workflow.compile(checkpointer=checkpointer)
