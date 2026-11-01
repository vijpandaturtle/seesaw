"""Each job writes its artifacts to its own folder.

The failure this guards: every job wrote outputs/research_plan.md and
outputs/results_bundle.json, so a later job overwrote an earlier one's plan
and results, and the earlier job's row pointed at someone else's files.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SEESAW_ARTIFACTS_DIR", raising=False)
    monkeypatch.setenv("SEESAW_DB_PATH", str(tmp_path / "jobs.db"))
    from shared.db import jobs as module

    monkeypatch.setattr(module, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(module, "LOGS_DIR", tmp_path / "job_logs")
    return module


def test_job_dir_is_per_job_and_follows_the_artifacts_volume(jobs, tmp_path, monkeypatch):
    assert jobs.job_dir("a") == tmp_path / "outputs" / "jobs" / "a"
    assert jobs.job_dir("a") != jobs.job_dir("b")
    monkeypatch.setenv("SEESAW_ARTIFACTS_DIR", "/artifacts")
    assert str(jobs.job_dir("a")) == "/artifacts/jobs/a"


def test_every_stage_is_given_the_jobs_folder(jobs, tmp_path, monkeypatch):
    from orchestrator.src import worker
    from orchestrator.src.clients import lens_client, quill_client, scout_client

    seen = {}

    def fake_scout(question, thread_id, output_dir):
        seen["scout"] = output_dir
        output_dir.mkdir(parents=True)
        (plan := output_dir / "research_plan.md").write_text("plan")
        return plan

    def fake_lens(plan, thread_id, output_dir):
        seen["lens"] = output_dir
        (path := output_dir / "results_bundle.json").write_text("{}")
        return {"n_success": 1, "n_total": 1}, path

    class Report:
        overall_assessment = "moderate"

    def fake_quill(bundle_path, thread_id, output_dir):
        seen["quill"] = output_dir
        return Report(), output_dir / "critique.md"

    monkeypatch.setattr(scout_client, "run_scout", fake_scout)
    monkeypatch.setattr(lens_client, "run_lens", fake_lens)
    monkeypatch.setattr(quill_client, "run_quill", fake_quill)

    job = jobs.create("q", "m", auto_approve=True)
    assert worker.run_job(job.id) == 0

    folder = tmp_path / "outputs" / "jobs" / job.id
    assert seen == {"scout": folder, "lens": folder, "quill": folder}
    after = jobs.get(job.id)
    assert after.plan_path == str(folder / "research_plan.md")
    assert after.bundle_path == str(folder / "results_bundle.json")
    assert after.report_path == str(folder / "critique.md")


def test_scout_saves_into_the_given_folder_only(tmp_path):
    from scout.mcp_server.src.tools import make_save_research_plan

    save = make_save_research_plan(tmp_path / "job")
    save.invoke({"plan": "# plan", "filename": "../../escape_research_plan.md"})
    assert (tmp_path / "job" / "escape_research_plan.md").read_text() == "# plan"
    assert not (tmp_path / "escape_research_plan.md").exists()


def test_lens_bundle_goes_to_the_states_folder(tmp_path):
    from lens.mcp_server.src.nodes.collect_results import collect_results

    collect_results({"research_question": "q", "model_name": "m", "results": [],
                     "output_dir": str(tmp_path / "job")})
    assert json.loads((tmp_path / "job" / "results_bundle.json").read_text())["n_total"] == 0


def test_lens_plots_go_to_the_jobs_folder_whatever_the_plan_says(tmp_path, monkeypatch):
    import importlib

    pytest.importorskip("transformer_lens")
    node = importlib.import_module("lens.mcp_server.src.nodes.run_experiment")
    seen = {}

    def tool(model, prompts, output_dir=None):
        seen["output_dir"] = output_dir
        return node.ExperimentResult(name="e", tool="t", model_name="m", prompts=prompts)

    monkeypatch.setenv("SEESAW_INTERP_BACKEND", "seesaw")
    monkeypatch.setattr(node, "get_model", lambda name: object())
    monkeypatch.setattr(node, "registry", lambda: {"t": tool})
    monkeypatch.setattr(node, "normalize_tool_kwargs", lambda tool, kw, n: (dict(kw), [], []))
    spec = {"name": "e", "tool": "t", "model_name": "m", "prompts": ["p"],
            "tool_kwargs": {"output_dir": "/etc"}}
    out = node.run_experiment({"experiment_queue": [spec], "output_dir": str(tmp_path / "job")})
    assert seen["output_dir"] == tmp_path / "job" / "plots"
    assert (tmp_path / "job" / "plots").is_dir()
    assert "output_dir" not in out["last_result"]["tool_kwargs"]
