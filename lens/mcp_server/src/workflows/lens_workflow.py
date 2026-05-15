"""LangGraph workflow for the Lens agent.

Graph structure:
    parse_plan → load_model → run_experiment → interpret_result → check_followup
                                    ▲                                      │
                                    └──────── more experiments? ───────────┘
                                                                           │
                                                                   collect_results → END
"""

import json

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from ..app.model_session import get_model
from ..app.sandbox import run_in_sandbox
from ..app.schemas import ExperimentResult
from ..config import (
    ANTHROPIC_API_KEY,
    LENS_LLM_MAX_TOKENS,
    LENS_LLM_MODEL,
    LENS_LLM_TEMPERATURE,
    MAX_FOLLOWUPS,
    OUTPUTS_DIR,
)
from ..tools import TOOL_REGISTRY


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


# ── Structured output models for Claude calls ─────────────────────────────────
class ExperimentSpecModel(BaseModel):
    name: str
    tool: str
    model_name: str
    prompts: list[str]
    what_to_measure: str
    hypothesis_tested: str
    expected_outcome: str
    tool_kwargs: dict = {}


class ParsedPlanModel(BaseModel):
    research_question: str
    model_name: str
    experiments: list[ExperimentSpecModel]


class FollowUpDecision(BaseModel):
    needs_followup: bool
    reason: str
    followup_tool: str | None = None
    followup_prompts: list[str] = []
    followup_description: str | None = None
    followup_kwargs: dict = {}


def _make_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=LENS_LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=LENS_LLM_TEMPERATURE,
        max_tokens=LENS_LLM_MAX_TOKENS,
    )


# ── Node 1: parse_plan ────────────────────────────────────────────────────────
def parse_plan(state: LensState) -> dict:
    print("📋 [parse_plan] Extracting experiments...")
    llm = _make_llm()
    structured_llm = llm.with_structured_output(ParsedPlanModel)
    prompt = (
        f"Extract the experiments from this Research Plan.\n"
        f"For each experiment, extract the tool name, model, prompts, and what to measure.\n"
        f"Only include experiments whose tool is one of: {list(TOOL_REGISTRY.keys())}.\n"
        f"For tools that need io_tokens and subject_tokens, include them in tool_kwargs.\n"
        f"If no specific prompts are given, generate 2-3 appropriate IOI-style prompts.\n\n"
        f"Research Plan:\n{state['research_plan']}"
    )
    parsed = structured_llm.invoke(prompt)
    queue  = [e.model_dump() for e in parsed.experiments]
    print(f"   Found {len(queue)} experiments: {[e['name'] for e in queue]}")
    return {
        "research_question": parsed.research_question,
        "model_name":        parsed.model_name,
        "experiment_queue":  queue,
        "results":           [],
        "followup_count":    0,
        "last_result":       None,
        "bundle":            None,
    }


# ── Node 2: load_model ────────────────────────────────────────────────────────
def load_model_node(state: LensState) -> dict:
    print(f"🧠 [load_model] {state['model_name']}")
    get_model(state["model_name"])
    return {}


# ── Node 3: run_experiment ────────────────────────────────────────────────────
def run_experiment(state: LensState) -> dict:
    queue = list(state["experiment_queue"])
    spec  = queue.pop(0)
    print(f"\n🔬 [run_experiment] '{spec['name']}' using {spec['tool']}")

    tool_fn = TOOL_REGISTRY.get(spec["tool"])
    if tool_fn is None:
        result = ExperimentResult(
            name=spec["name"],
            tool=spec["tool"],
            model_name=spec["model_name"],
            prompts=spec["prompts"],
            status="failed",
            error=f"Tool '{spec['tool']}' not in registry: {list(TOOL_REGISTRY.keys())}",
        )
    else:
        m      = get_model(spec["model_name"])
        result = run_in_sandbox(
            tool_fn,
            tool_kwargs={"model": m, "prompts": spec["prompts"], **spec.get("tool_kwargs", {})},
            experiment_name=spec["name"],
        )

    icon = "✅" if result.status == "success" else "❌"
    print(f"   {icon} {result.status} | plots={len(result.plot_paths)}")
    return {
        "experiment_queue": queue,
        "last_result": result.__dict__ | {"plot_paths": [str(p) for p in result.plot_paths]},
    }


# ── Node 4: interpret_result ──────────────────────────────────────────────────
def interpret_result(state: LensState) -> dict:
    result = state["last_result"]
    print(f"💬 [interpret_result] '{result['name']}'")

    if result["status"] != "success":
        result["summary"] = f"Experiment failed: {result.get('error')}"
        return {"results": list(state["results"]) + [result], "last_result": result}

    llm          = _make_llm()
    data_preview = json.dumps(result["data"], indent=2)[:2_000]
    response     = llm.invoke(
        f"You are an AI safety researcher specialised in mechanistic interpretability.\n"
        f"Interpret this experiment result in 2-3 paragraphs:\n"
        f"1. What do the results reveal about the model's internal mechanisms?\n"
        f"2. What is the most important finding?\n"
        f"3. How does this connect to the research question: \"{state['research_question']}\"\n\n"
        f"Experiment: {result['name']} | Tool: {result['tool']}\n"
        f"Data: {data_preview}"
    )
    result["summary"] = response.content
    return {"results": list(state["results"]) + [result], "last_result": result}


# ── Node 5: check_followup ────────────────────────────────────────────────────
def check_followup(state: LensState) -> dict:
    result         = state["last_result"]
    followup_count = state.get("followup_count", 0)
    if followup_count >= MAX_FOLLOWUPS or result["status"] != "success":
        return {}

    llm            = _make_llm()
    structured_llm = llm.with_structured_output(FollowUpDecision)
    decision       = structured_llm.invoke(
        f"Based on this result, decide if an immediate follow-up is needed.\n"
        f"Be conservative — only if the result reveals something a different tool can directly test.\n"
        f"Available tools: {list(TOOL_REGISTRY.keys())}\n"
        f"Follow-ups used: {followup_count}/{MAX_FOLLOWUPS}\n"
        f"Summary: {result['summary']}"
    )

    if decision.needs_followup and decision.followup_tool:
        new_spec = {
            "name":             f"Follow-up: {decision.followup_description or decision.followup_tool}",
            "tool":             decision.followup_tool,
            "model_name":       result["model_name"],
            "prompts":          decision.followup_prompts or result["prompts"],
            "what_to_measure":  decision.followup_description or "",
            "hypothesis_tested": "follow-up",
            "expected_outcome": "",
            "tool_kwargs":      decision.followup_kwargs,
        }
        print(f"   ➕ Follow-up: {new_spec['name']}")
        return {
            "experiment_queue": [new_spec] + list(state["experiment_queue"]),
            "followup_count":   followup_count + 1,
        }
    return {}


# ── Node 6: collect_results ───────────────────────────────────────────────────
def collect_results(state: LensState) -> dict:
    print("\n📦 [collect_results] Bundling...")
    bundle = {
        "research_question": state["research_question"],
        "model_name":        state["model_name"],
        "results":           state["results"],
        "n_total":           len(state["results"]),
        "n_success":         sum(1 for r in state["results"] if r["status"] == "success"),
        "n_failed":          sum(1 for r in state["results"] if r["status"] != "success"),
    }
    bundle_path = OUTPUTS_DIR / "results_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, default=str))
    print(f"   ✅ {bundle['n_success']}/{bundle['n_total']} succeeded → {bundle_path}")
    return {"bundle": bundle}


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
