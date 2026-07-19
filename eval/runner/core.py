"""Task runner core: execute a Task's agents and capture everything graders need.

run_task(task) is the entry point:
    1. dispatch on task.target (scout | lens | quill | pipeline) → an AgentRun
       carrying artifacts (plan / bundle / critique), trajectory, usage, timing
    2. walk task.graders, dispatching each GraderSpec.type to its handler
       (see handlers.py) → list[GraderResult]
    3. collect task.tracked_metrics from the AgentRun
    4. return (and optionally persist) a TaskResult

Agents are invoked through their real LangGraph graphs (not the orchestrator
clients) so we can capture per-message token usage and tool-call trajectory.
Agent packages import torch/langgraph — all agent imports are lazy so that
loading this module (or grading a canned run) needs neither.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..graders import GraderResult
from ..tasks import Task

RUNS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "eval_runs"


# ── Run record ───────────────────────────────────────────────────────────────
@dataclass
class AgentRun:
    """Everything a grader may need about one task execution."""
    target: str
    question: str | None = None
    # artifacts
    plan_text: str | None = None
    plan_path: Path | None = None
    bundle: dict | None = None            # ExperimentBundle as a plain dict
    bundle_path: Path | None = None
    report: Any | None = None             # CritiqueReport dataclass (or dict)
    report_path: Path | None = None
    # observability
    trajectory: list[dict] = field(default_factory=list)   # [{"name","args"}]
    usage: dict = field(default_factory=dict)              # token/turn counts
    timing: dict = field(default_factory=dict)             # start/first_token/end
    errors: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    task_id: str
    target: str
    grader_results: list[GraderResult]
    metrics: dict
    started_at: float
    finished_at: float
    run: AgentRun

    @property
    def scores(self) -> dict[str, float | None]:
        return {r.criterion: r.score for r in self.grader_results}

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "target": self.target,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "scores": self.scores,
            "grader_results": [dataclasses.asdict(r) for r in self.grader_results],
            "metrics": self.metrics,
            "trajectory": self.run.trajectory,
            "usage": self.run.usage,
            "errors": self.run.errors,
            "artifacts": {
                "plan_path": str(self.run.plan_path) if self.run.plan_path else None,
                "bundle_path": str(self.run.bundle_path) if self.run.bundle_path else None,
                "report_path": str(self.run.report_path) if self.run.report_path else None,
            },
        }


# ── Instrumented agent adapters ──────────────────────────────────────────────
def _accumulate_usage(msg, usage: dict) -> None:
    meta = getattr(msg, "usage_metadata", None)
    if meta:
        usage["n_input_tokens"] = usage.get("n_input_tokens", 0) + meta.get("input_tokens", 0)
        usage["n_output_tokens"] = usage.get("n_output_tokens", 0) + meta.get("output_tokens", 0)
        usage["n_total_tokens"] = usage.get("n_input_tokens", 0) + usage.get("n_output_tokens", 0)


def run_scout_instrumented(question: str, run: AgentRun) -> None:
    """Scout ReAct agent with trajectory + usage capture."""
    from scout.mcp_server.src.app.agent import build_agent
    from scout.mcp_server.src.config.settings import OUTPUTS_DIR

    agent = build_agent()
    config = {"configurable": {"thread_id": f"eval-{int(time.time())}"}}
    start = time.time()
    run.timing["start"] = start
    seen_ids: set[str] = set()

    for chunk in agent.stream({"messages": [{"role": "user", "content": question}]},
                              config=config, stream_mode="values"):
        run.timing.setdefault("first_token", time.time())
        for msg in chunk["messages"]:
            mid = getattr(msg, "id", None)
            if mid in seen_ids:
                continue
            if mid:
                seen_ids.add(mid)
            for tc in getattr(msg, "tool_calls", None) or []:
                run.trajectory.append({"name": tc.get("name"), "args": tc.get("args", {})})
            if msg.__class__.__name__ == "AIMessage":
                run.usage["n_turns"] = run.usage.get("n_turns", 0) + 1
                _accumulate_usage(msg, run.usage)

    run.timing["end"] = time.time()
    run.usage["n_toolcalls"] = len(run.trajectory)

    candidates = [p for p in OUTPUTS_DIR.glob("*research_plan*.md") if p.stat().st_mtime >= start]
    if candidates:
        run.plan_path = max(candidates, key=lambda p: p.stat().st_mtime)
        run.plan_text = run.plan_path.read_text()
    else:
        run.errors.append("scout finished but saved no *research_plan*.md")


def run_lens_instrumented(plan_text: str, run: AgentRun) -> None:
    """Lens StateGraph; trajectory reconstructed from the executed experiments."""
    from lens.mcp_server.src.config.settings import OUTPUTS_DIR
    from lens.mcp_server.src.workflows.lens_workflow import build_lens_graph

    graph = build_lens_graph()
    run.timing.setdefault("start", time.time())
    final = graph.invoke({"research_plan": plan_text},
                         config={"configurable": {"thread_id": f"eval-{int(time.time())}"}})
    run.timing["end"] = time.time()

    bundle = final.get("bundle")
    if bundle is None:
        run.errors.append("lens finished but produced no bundle")
        return
    run.bundle = bundle
    run.bundle_path = OUTPUTS_DIR / "results_bundle.json"
    for r in bundle.get("results", []):
        run.trajectory.append({"name": r.get("tool"), "args": {"prompts": r.get("prompts", [])}})
    run.usage["n_toolcalls"] = run.usage.get("n_toolcalls", 0) + len(bundle.get("results", []))


def run_quill_instrumented(bundle_dict: dict, run: AgentRun) -> None:
    """Quill critique workflow on an in-memory bundle (canned or from Lens)."""
    from quill.mcp_server.src.config.settings import OUTPUTS_DIR
    from quill.mcp_server.src.models.schemas import ExperimentBundle, ExperimentResult
    from quill.mcp_server.src.workflows.critique_workflow import build_quill_graph

    bundle = ExperimentBundle(
        research_question=bundle_dict["research_question"],
        model_name=bundle_dict["model_name"],
        results=[ExperimentResult(**r) for r in bundle_dict["results"]],
    )
    graph = build_quill_graph()
    run.timing.setdefault("start", time.time())
    final = graph.invoke({"bundle": bundle, "output_dir": OUTPUTS_DIR},
                         config={"configurable": {"thread_id": f"eval-{int(time.time())}"}})
    run.timing["end"] = time.time()

    run.report = final.get("critique_report")
    run.report_path = final.get("critique_path")
    run.bundle = run.bundle or bundle_dict
    if run.report is None:
        run.errors.append("quill finished but produced no critique report")


def _synthesize_plan(task: Task) -> str:
    """Minimal plan for target:lens tasks (no Scout in the loop). Honors
    plan_constraint.allowed_tools so traps like the causal-language task hold."""
    allowed = (task.extra.get("plan_constraint") or {}).get("allowed_tools")
    tools_line = f"Use ONLY these tools: {', '.join(allowed)}." if allowed else ""
    return (
        f"# Research Plan\n\n## Research Question\n{task.question}\n\n"
        f"## Target Model\n{task.extra.get('model', 'gpt2')}\n\n"
        f"## Experiments\n{tools_line}\n"
        f"Run the appropriate experiments to answer the question, with prompts "
        f"suited to the behaviour under study.\n"
    )


def execute_agents(task: Task) -> AgentRun:
    """Dispatch on task.target and return the populated AgentRun."""
    run = AgentRun(target=task.target or "pipeline", question=task.question)
    if run.target == "scout":
        run_scout_instrumented(task.question, run)
    elif run.target == "lens":
        run_lens_instrumented(_synthesize_plan(task), run)
    elif run.target == "quill":
        bundle = task.extra.get("input_bundle")
        if not bundle:
            raise ValueError(f"task {task.id}: target=quill requires input_bundle")
        run_quill_instrumented(bundle, run)
    elif run.target == "pipeline":
        run_scout_instrumented(task.question, run)
        if run.plan_text:
            run_lens_instrumented(run.plan_text, run)
        if run.bundle:
            run_quill_instrumented(run.bundle, run)
    else:
        raise ValueError(f"task {task.id}: unknown target {run.target!r}")
    return run


# ── Metric collection ────────────────────────────────────────────────────────
def collect_metrics(task: Task, run: AgentRun) -> dict:
    t = run.timing
    derived = {
        "n_turns": run.usage.get("n_turns"),
        "n_toolcalls": run.usage.get("n_toolcalls"),
        "n_total_tokens": run.usage.get("n_total_tokens"),
        "n_input_tokens": run.usage.get("n_input_tokens"),
        "n_output_tokens": run.usage.get("n_output_tokens"),
        "time_to_first_token": (t["first_token"] - t["start"]) if {"first_token", "start"} <= t.keys() else None,
        "time_to_last_token": (t["end"] - t["start"]) if {"end", "start"} <= t.keys() else None,
        "wall_time": (t["end"] - t["start"]) if {"end", "start"} <= t.keys() else None,
        "output_tokens_per_sec": None,
    }
    if derived["n_output_tokens"] and derived["wall_time"]:
        derived["output_tokens_per_sec"] = derived["n_output_tokens"] / derived["wall_time"]

    out: dict[str, dict] = {}
    for tm in task.tracked_metrics:
        out[tm.type] = {name: derived.get(name) for name in tm.metrics}
    return out


# ── Entry point ──────────────────────────────────────────────────────────────
def run_task(task: Task, *, persist: bool = True) -> TaskResult:
    from .handlers import dispatch_grader   # local import to avoid cycles

    started = time.time()
    run = execute_agents(task)
    results: list[GraderResult] = []
    for spec in task.graders:
        results.extend(dispatch_grader(spec, task, run))
    metrics = collect_metrics(task, run)
    finished = time.time()

    result = TaskResult(task_id=task.id, target=run.target, grader_results=results,
                        metrics=metrics, started_at=started, finished_at=finished, run=run)
    if persist:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = RUNS_DIR / f"{task.id}_{int(started)}.json"
        out.write_text(json.dumps(result.to_dict(), indent=2, default=str))
    return result
