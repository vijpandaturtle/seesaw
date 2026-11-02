"""Pathways inside jobs: the catalog's checks, and how the worker runs one.

What has to hold:
  * malformed inputs are named precisely, before anything runs
  * Lens's stage runs a job's pathway on its approved inputs and writes the
    pathway's report and a JSON result (plain values) next to Lens's bundle
  * a pathway that raises fails the stage, where retry can re-run it
  * standalone pathway jobs (one stage, hand-entered inputs) are retired:
    the rows already stored still read, and one still queued fails with a
    message saying what to do instead

The pathway itself is faked here; tests/test_pathways_gpt2.py checks the real
ones against known answers, and tests/test_pathway_research.py covers Scout
choosing the inputs and reruns.
"""

from __future__ import annotations

import json

import numpy as np
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
    monkeypatch.setattr(catalog, "_tokenizer", lambda model: None)   # skip the tokenizer check
    return module


@pytest.fixture
def at_lens(jobs, tmp_path, monkeypatch):
    """A job queued at Lens with approved IOI inputs (what Scout's gate, or a
    rerun, leaves), and a fake Lens with no extras of its own."""
    from orchestrator.src.clients import lens_client

    def fake_lens(research_plan, thread_id, output_dir):
        bundle = {"research_question": "IOI", "model_name": "gpt2", "results": [],
                  "n_total": 0, "n_success": 0, "n_failed": 0}
        (path := output_dir / "results_bundle.json").write_text(json.dumps(bundle))
        return bundle, path

    monkeypatch.setattr(lens_client, "run_lens", fake_lens)
    (plan := tmp_path / "plan.md").write_text("# Research Plan: IOI\n")
    job = jobs.create("IOI circuit", model="gpt2", params={"pathway": "circuit_discovery"})
    jobs.update(job.id, stage="lens", plan_path=str(plan),
                params=json.dumps({"pathway": "circuit_discovery", "pathway_params": IOI}))
    return job.id


# ── catalog ──────────────────────────────────────────────────────────────────
def test_valid_inputs_pass():
    assert catalog.validate("circuit_discovery", IOI) == []


def test_unknown_pathway_is_refused():
    assert "unknown pathway" in catalog.validate("nope", IOI)[0]


def test_missing_and_misaligned_inputs_are_named():
    problems = catalog.validate("attribute_audit", {**IOI, "negative_tokens": [" a", " b"]})
    assert any("counterfactual_prompts" in p for p in problems)
    assert any("negative_tokens has 2 entries" in p for p in problems)


def test_unexpected_inputs_are_refused():
    problems = catalog.validate("feature_explain", {**IOI, "corrupted_prompts": ["x"]})
    assert problems == ["feature_explain does not take 'corrupted_prompts'"]


def test_feature_explain_needs_its_sae_model():
    assert catalog.validate("feature_explain", IOI, model="pythia-70m")
    assert catalog.validate("feature_explain", IOI, model="gpt2") == []


# ── worker ───────────────────────────────────────────────────────────────────
def test_lens_runs_the_pathway_and_writes_its_result(jobs, at_lens, monkeypatch):
    from orchestrator.src import worker

    seen = {}

    def fake_run(name, model_name, params, output_dir):
        seen.update(name=name, model=model_name, params=params, out=output_dir)
        (plot := output_dir / "dla.png").write_bytes(b"png")
        result = PathwayResult(pathway=name, verdict="circuit_found",
                               evidence=["L9H9 carries 40% of the logit diff."],
                               findings={"baseline_logit_diff": np.float32(3.5)})
        result.steps["dla"] = type("R", (), {"plot_paths": [plot]})()
        return result

    monkeypatch.setattr(catalog, "run_from_params", fake_run)
    assert worker.run_job(at_lens) == 0
    assert jobs.get(at_lens).stage == "quill"
    assert seen["name"] == "circuit_discovery" and seen["params"] == IOI
    out = jobs.job_dir(at_lens) / "pathway"
    assert seen["out"] == out

    result = json.loads((out / "result.json").read_text())
    assert result["verdict"] == "circuit_found"
    assert result["findings"] == {"baseline_logit_diff": 3.5}
    assert result["steps"] == ["dla"]
    assert result["plot_paths"][0].endswith("dla.png")
    assert "circuit_found" in (out / "report.md").read_text()


def test_a_failing_pathway_fails_the_lens_stage(jobs, at_lens, monkeypatch):
    from orchestrator.src import worker

    def boom(*_args):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(catalog, "run_from_params", boom)
    assert worker.run_job(at_lens) == 1
    failed = jobs.get(at_lens)
    assert (failed.status, failed.stage) == (jobs.FAILED, "lens")
    assert "lens: RuntimeError: CUDA out of memory" in failed.error


# ── retired standalone pathway jobs ──────────────────────────────────────────
def legacy_job(jobs, status, stage):
    """A standalone pathway job row, as stored before they were retired."""
    with jobs.connect() as conn:
        jobs._ex(conn, "INSERT INTO jobs (id, question, model, status, stage, auto_approve,"
                       " created_at, updated_at, kind, params) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 ("legacy01", "IOI circuit", "gpt2", status, stage, 0, 1.0, 1.0, "pathway",
                  json.dumps({"pathway": "circuit_discovery", **IOI})))
    return jobs.get("legacy01")


def test_stored_pathway_jobs_still_read(jobs):
    job = legacy_job(jobs, jobs.DONE, "done")
    assert (job.kind, job.stages, job.completed_stages) == ("pathway", ("pathway",), ["pathway"])
    assert job.params_dict["pathway"] == "circuit_discovery"


def test_a_queued_pathway_job_fails_saying_what_to_do_instead(jobs):
    from orchestrator.src import worker

    legacy_job(jobs, jobs.QUEUED, "pathway")
    assert worker.run_job("legacy01") == 1
    failed = jobs.get("legacy01")
    assert failed.status == jobs.FAILED
    assert "a pathway runs inside a research job" in failed.error


def test_hand_entered_inputs_cant_start_a_job(jobs):
    with pytest.raises(ValueError, match="name only a pathway"):
        jobs.create("x", params={"pathway": "circuit_discovery", **IOI})
    with pytest.raises(TypeError):
        jobs.create("x", kind="pathway", params={"pathway": "circuit_discovery", **IOI})
    assert jobs.list_jobs() == []
