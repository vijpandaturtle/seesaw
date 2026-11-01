"""Pathway jobs: a hand-coded pathway queued and run like any other job.

What has to hold:
  * malformed inputs are refused when the job is created, not minutes later
    in the worker
  * a pathway job is one stage with no human gate
  * the worker writes the pathway's report and a JSON result the dashboard
    reads, and finishes the job

The pathway itself is faked here; tests/test_pathways_gpt2.py checks the real
ones against known answers.
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
    return module


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


# ── store ────────────────────────────────────────────────────────────────────
def test_pathway_job_is_one_stage_with_no_gate(jobs):
    job = jobs.create("IOI circuit", kind="pathway", params={"pathway": "circuit_discovery", **IOI})
    assert (job.kind, job.stage, job.status) == ("pathway", "pathway", jobs.QUEUED)
    assert jobs.get(job.id).params_dict == {"pathway": "circuit_discovery", **IOI}

    jobs.advance(jobs.get(job.id))
    after = jobs.get(job.id)
    assert (after.stage, after.status) == ("done", jobs.DONE)
    assert after.completed_stages == ["pathway"]


def test_bad_pathway_inputs_never_reach_the_store(jobs):
    with pytest.raises(ValueError, match="positive_tokens"):
        jobs.create("x", kind="pathway", params={"pathway": "circuit_discovery", "prompts": ["p"]})
    with pytest.raises(ValueError, match="unknown job kind"):
        jobs.create("x", kind="survey")
    with pytest.raises(ValueError, match="only pathway jobs"):
        jobs.create("x", params={"prompts": ["p"]})
    assert jobs.list_jobs() == []


def test_research_jobs_are_unchanged(jobs):
    job = jobs.get(jobs.create("q").id)
    assert (job.kind, job.stage, job.params) == ("research", "scout", None)


# ── worker ───────────────────────────────────────────────────────────────────
def test_worker_runs_the_pathway_and_writes_its_result(jobs, monkeypatch):
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
    job = jobs.create("IOI circuit", model="gpt2", kind="pathway",
                      params={"pathway": "circuit_discovery", **IOI})

    assert worker.run_job(job.id) == 0
    done = jobs.get(job.id)
    assert done.status == jobs.DONE
    assert seen["name"] == "circuit_discovery" and seen["params"] == IOI
    assert seen["out"] == jobs.job_dir(job.id) / "pathway"

    result = json.loads(open(done.bundle_path).read())
    assert result["verdict"] == "circuit_found"
    assert result["findings"] == {"baseline_logit_diff": 3.5}
    assert result["steps"] == ["dla"]
    assert result["plot_paths"][0].endswith("dla.png")
    assert "circuit_found" in open(done.report_path).read()


def test_a_failing_pathway_fails_the_job(jobs, monkeypatch):
    from orchestrator.src import worker

    def boom(*_args):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(catalog, "run_from_params", boom)
    job = jobs.create("x", kind="pathway", params={"pathway": "circuit_discovery", **IOI})
    assert worker.run_job(job.id) == 1
    failed = jobs.get(job.id)
    assert failed.status == jobs.FAILED
    assert "pathway: RuntimeError: CUDA out of memory" in failed.error
