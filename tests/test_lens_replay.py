"""Lens replaying experiments it ran before, for a rerun.

What has to hold:
  * given `replay`, Lens runs exactly those experiments, in order, with the
    tools, prompts and arguments given: no LLM reads the plan
  * no follow-ups are added (the source's follow-ups are already in the list)
  * the bundle has one result per replayed experiment

The experiment runner and the interpreter are faked; nothing loads a model or
calls an LLM, which is the point: a replay that called one would fail here.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("transformer_lens")  # the workflow imports Lens's tools

from lens.mcp_server.src.workflows import lens_workflow  # noqa: E402

IOI = ["When Mary and John went to the store, John gave a drink to"]
EXPERIMENTS = [
    {"name": "logit_lens_ioi", "tool": "logit_lens", "model_name": "pythia-70m",
     "prompts": IOI, "tool_kwargs": {}},
    {"name": "Follow-up: patch L9H9", "tool": "activation_patching", "model_name": "pythia-70m",
     "prompts": IOI, "tool_kwargs": {"positive_tokens": [" Mary"], "negative_tokens": [" John"]}},
]


def test_a_replay_runs_the_given_experiments_without_reading_the_plan(tmp_path, monkeypatch):
    ran = []

    def no_parse(state):
        raise AssertionError("a replay must not read the plan")

    def fake_run(state):
        queue = list(state["experiment_queue"])
        spec = queue.pop(0)
        ran.append(spec)
        result = {**spec, "status": "success", "summary": "", "data": {}, "plot_paths": [],
                  "error": None}
        return {"experiment_queue": queue, "last_result": result}

    def fake_interpret(state):
        return {"results": list(state["results"]) + [state["last_result"]]}

    monkeypatch.setattr(lens_workflow, "parse_plan", no_parse)
    monkeypatch.setattr(lens_workflow, "load_model_node", lambda state: {})
    monkeypatch.setattr(lens_workflow, "run_experiment", fake_run)
    monkeypatch.setattr(lens_workflow, "interpret_result", fake_interpret)

    final = lens_workflow.build_lens_graph().invoke(
        {"research_plan": "# a plan nobody reads", "output_dir": str(tmp_path),
         "replay": {"research_question": "IOI", "model_name": "pythia-70m",
                    "experiments": EXPERIMENTS}},
        config={"configurable": {"thread_id": "replay-test"}},
    )

    assert ran == EXPERIMENTS                       # exactly these, in order, nothing added
    bundle = json.loads((tmp_path / "results_bundle.json").read_text())
    assert (bundle["research_question"], bundle["model_name"]) == ("IOI", "pythia-70m")
    assert [r["name"] for r in bundle["results"]] == [e["name"] for e in EXPERIMENTS]
    assert final["bundle"]["n_success"] == 2
