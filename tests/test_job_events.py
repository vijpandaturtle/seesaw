"""A job's timeline: what its worker did, recorded as it happens.

What has to hold:
  * nothing is recorded outside a worker (no job bound), so the agents run
    unchanged in tests, notebooks and the MCP servers
  * spans record how long a block took and how it ended, and events nest
    under the span open when they started
  * a LangGraph run's nodes, LLM calls (model, tokens, answer) and tool calls
    (input, output) land on the timeline, each under its node
  * a worker records each stage, each pathway step, each failed input check,
    the error behind a failed stage, and every change of job status
  * long values are cut short, and a timeline that can't be written never
    fails the job

LLMs are faked with LangChain's fake chat model; nothing calls a real one.
"""

from __future__ import annotations

import json

import pytest

from seesaw.pathways import catalog
from seesaw.pathways.result import PathwayResult

IOI = {
    "prompts": ["When Mary and John went to the store, John gave a drink to"],
    "positive_tokens": [" Mary"],
    "negative_tokens": [" John"],
}


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SEESAW_ARTIFACTS_DIR", raising=False)
    monkeypatch.setenv("SEESAW_DB_PATH", str(tmp_path / "jobs.db"))
    from shared.db import jobs as module

    monkeypatch.setattr(module, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(module, "LOGS_DIR", tmp_path / "job_logs")
    return module


@pytest.fixture
def events(jobs):
    from shared.db import events as module

    yield module
    module._open.clear()


def by_name(rows):
    return {e["name"]: e for e in rows}


# ── recording ────────────────────────────────────────────────────────────────
def test_nothing_is_recorded_without_a_bound_job(jobs, events):
    job = jobs.create("q")
    assert events.emit(events.STATUS, "done") is None
    with events.span(events.STAGE, "scout") as s:
        assert s.id is None
    assert events.callbacks() == []
    assert events.list_events(job.id) == []


def test_spans_time_a_block_and_nest_what_happens_inside(jobs, events):
    job = jobs.create("q")
    with events.bind(job.id, "lens"):
        with events.span(events.STAGE, "lens") as stage:
            events.emit(events.EXPERIMENT, "ioi_ablation", summary="success")
            stage.summary = "1 experiment"
        events.emit(events.STATUS, "done")

    rows = by_name(events.list_events(job.id))
    assert rows["lens"]["status"] == "ok"
    assert rows["lens"]["summary"] == "1 experiment"
    assert rows["lens"]["duration_ms"] >= 0
    assert rows["ioi_ablation"]["parent_id"] == rows["lens"]["id"]
    assert rows["ioi_ablation"]["stage"] == "lens"
    assert rows["done"]["parent_id"] is None


def test_a_span_that_raises_is_marked_failed_and_reraises(jobs, events):
    job = jobs.create("q")
    with events.bind(job.id, "scout"), pytest.raises(ValueError):
        with events.span(events.STAGE, "scout"):
            raise ValueError("no plan saved")

    (row,) = events.list_events(job.id)
    assert row["status"] == "error"
    assert row["summary"] == "ValueError: no plan saved"


def test_long_values_are_cut_short(jobs, events):
    job = jobs.create("q")
    with events.bind(job.id):
        events.emit(events.LLM, "m", data={"response": "x" * 10_000, "ids": list(range(500))})

    (row,) = events.list_events(job.id)
    assert len(row["data"]["response"]) < 4_100
    assert row["data"]["response"].endswith("more chars]")
    assert len(row["data"]["ids"]) == events.MAX_ITEMS + 1


def test_a_store_that_cant_be_written_never_fails_the_job(jobs, events, monkeypatch, capsys):
    def broken():
        raise RuntimeError("database is down")

    monkeypatch.setattr(jobs, "connect", broken)
    monkeypatch.setattr(events, "_warned", False)
    with events.bind("abc12345", "lens"):
        assert events.emit(events.STATUS, "done") is None
        with events.span(events.STAGE, "lens"):
            pass
    assert "couldn't write to the job timeline" in capsys.readouterr().out


def test_deleting_a_job_deletes_its_events(jobs, events):
    job = jobs.create("q")
    with events.bind(job.id):
        events.emit(events.STATUS, "done")
    jobs.delete(job.id)
    assert events.list_events(job.id) == []


# ── LangGraph ────────────────────────────────────────────────────────────────
def test_nodes_llm_calls_and_tool_calls_are_recorded(jobs, events):
    from typing import TypedDict

    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph

    llm = GenericFakeChatModel(messages=iter([AIMessage(
        content="Heads L9H6 and L9H9 move the name.",
        usage_metadata={"input_tokens": 120, "output_tokens": 9, "total_tokens": 129})]))

    @tool
    def search_papers(query: str) -> str:
        """Search for papers."""
        return f"3 papers on {query}"

    class State(TypedDict):
        answer: str

    def think(state):
        found = search_papers.invoke({"query": "IOI"})
        return {"answer": llm.invoke(f"Which heads? {found}").content}

    graph = StateGraph(State)
    graph.add_node("think", think)
    graph.add_edge(START, "think")
    graph.add_edge("think", END)

    job = jobs.create("q")
    with events.bind(job.id, "scout"):
        out = graph.compile().invoke({"answer": ""}, config={"callbacks": events.callbacks()})
    assert out["answer"].startswith("Heads")

    rows = events.list_events(job.id)
    node = next(e for e in rows if e["kind"] == "node")
    llm_call = next(e for e in rows if e["kind"] == "llm")
    tool_call = next(e for e in rows if e["kind"] == "tool")

    assert (node["name"], node["status"]) == ("think", "ok")
    assert llm_call["parent_id"] == node["id"] and tool_call["parent_id"] == node["id"]
    assert llm_call["status"] == "ok"
    assert "120 in / 9 out tokens" in llm_call["summary"]
    assert llm_call["data"]["response"] == "Heads L9H6 and L9H9 move the name."
    assert "3 papers on IOI" in llm_call["data"]["prompt"]
    assert tool_call["name"] == "search_papers"
    assert tool_call["data"]["output"] == "3 papers on IOI"
    assert tool_call["data"]["input"] == {"query": "IOI"}


# ── worker ───────────────────────────────────────────────────────────────────
@pytest.fixture
def at_lens(jobs, tmp_path, monkeypatch):
    """A job queued at Lens with approved IOI inputs, and a fake Lens."""
    from orchestrator.src.clients import lens_client

    def fake_lens(research_plan, thread_id, output_dir):
        bundle = {"research_question": "IOI", "model_name": "gpt2", "results": [],
                  "n_total": 0, "n_success": 0, "n_failed": 0}
        (path := output_dir / "results_bundle.json").write_text(json.dumps(bundle))
        return bundle, path

    monkeypatch.setattr(lens_client, "run_lens", fake_lens)
    monkeypatch.setattr(catalog, "_tokenizer", lambda model: None)
    (plan := tmp_path / "plan.md").write_text("# Research Plan: IOI\n")
    job = jobs.create("IOI", params={"pathway": "circuit_discovery"})
    jobs.update(job.id, stage="lens", plan_path=str(plan),
                params=json.dumps({"pathway": "circuit_discovery", "pathway_params": IOI}))
    return job.id


def test_lens_records_its_stage_the_pathways_steps_and_status(jobs, events, at_lens, monkeypatch):
    from orchestrator.src import worker

    def fake_run(name, model_name, params, output_dir):
        result = PathwayResult(pathway=name, verdict="circuit_found")
        step = type("R", (), {"plot_paths": [], "data": {"logit_diff": 3.51, "top": ["L9H9"]}})()
        result.steps["direct_logit_attribution"] = step
        result.steps["head_patching"] = type("R", (), {"plot_paths": [], "data": {}})()
        return result

    monkeypatch.setattr(catalog, "run_from_params", fake_run)
    assert worker.run_job(at_lens) == 0

    rows = events.list_events(at_lens)
    assert [e["kind"] for e in rows] == ["stage", "pathway", "step", "step", "status"]
    stage, pathway, dla, patching, status = rows
    assert (stage["name"], stage["status"]) == ("lens", "ok")
    assert pathway["parent_id"] == stage["id"]
    assert pathway["summary"] == "verdict: circuit_found"
    assert dla["parent_id"] == patching["parent_id"] == pathway["id"]
    assert dla["name"] == "direct_logit_attribution"
    assert dla["summary"] == "logit_diff=3.51"
    assert dla["data"]["data"]["top"] == ["L9H9"]
    assert (status["name"], status["summary"]) == ("awaiting_approval",
                                                   "waiting for approval before quill")


def test_a_failed_stage_records_its_error_and_traceback(jobs, events, at_lens, monkeypatch):
    from orchestrator.src import worker

    def boom(*_args):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(catalog, "run_from_params", boom)
    assert worker.run_job(at_lens) == 1

    rows = by_name(events.list_events(at_lens))
    assert rows["lens"]["status"] == "error"
    assert rows["circuit_discovery"]["status"] == "error"
    assert rows["RuntimeError"]["summary"] == "CUDA out of memory"
    assert "Traceback" in rows["RuntimeError"]["data"]["traceback"]


def test_scouts_failed_input_checks_are_recorded(jobs, events, monkeypatch):
    from orchestrator.src import worker
    from orchestrator.src.clients import scout_client

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [0]

        def convert_ids_to_tokens(self, ids):
            return ["t0"]

    def plan(params):
        return ("# Research Plan\n\n```json\n"
                + json.dumps({"pathway": "circuit_discovery", "params": params}) + "\n```\n")

    plans = [plan({**IOI, "negative_tokens": []}), plan(IOI)]

    class FakeAgent:
        def __init__(self, output_dir):
            self.output_dir = output_dir

        def stream(self, payload, config, stream_mode):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "ioi_research_plan.md").write_text(plans.pop(0))
            return iter(())

    monkeypatch.setattr(catalog, "_tokenizer", lambda model: FakeTokenizer())
    monkeypatch.setattr(scout_client, "build_agent", lambda output_dir, search=True: FakeAgent(output_dir))
    job = jobs.create("IOI", params={"pathway": "circuit_discovery"})
    assert worker.run_job(job.id) == 0

    checks = [e for e in events.list_events(job.id) if e["kind"] == "check"]
    assert [c["status"] for c in checks] == ["error", "ok"]
    assert "asking Scout to revise (1/2)" in checks[0]["summary"]
    assert any("negative_tokens" in err for err in checks[0]["data"]["errors"])
    status = events.list_events(job.id)[-1]
    assert (status["name"], status["summary"]) == ("awaiting_approval",
                                                   "waiting for approval before lens")
