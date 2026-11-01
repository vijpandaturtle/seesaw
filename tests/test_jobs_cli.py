"""`seesaw jobs …`: the terminal drives the same job store the dashboard reads.

Runs against a throwaway SQLite store, with the local runner's process launch
recorded rather than performed, so no agent or model ever runs. What's checked
is the contract with the store: what gets queued, which transitions are
guarded, and what a follower prints.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MODAL_RUN_STAGE_URL", raising=False)
    monkeypatch.setenv("SEESAW_DB_PATH", str(tmp_path / "jobs.db"))
    from shared.db import jobs as module

    monkeypatch.setattr(module, "LOGS_DIR", tmp_path / "job_logs")
    return module


@pytest.fixture
def launched(monkeypatch):
    """Worker launches the local runner attempted, instead of real processes."""
    calls = []

    class FakePopen:
        pid = 4242

        def __init__(self, cmd, **kw):
            calls.append((cmd, kw))

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    return calls


def seesaw(monkeypatch, *argv) -> int:
    from seesaw import cli

    monkeypatch.setattr(cli, "load_dotenv", lambda: None)   # keep a real .env out of the test
    monkeypatch.setattr(sys, "argv", ["seesaw", *argv])
    return cli.main()


def test_submit_queues_a_job_and_starts_a_local_worker(jobs, launched, monkeypatch, capsys):
    assert seesaw(monkeypatch, "jobs", "submit", "which heads do IOI?", "--runner", "local") == 0
    [job] = jobs.list_jobs()
    assert (job.status, job.stage, job.model) == (jobs.QUEUED, "scout", "gpt2")
    [(cmd, kw)] = launched
    assert cmd[-3:] == ["-m", "orchestrator.src.worker", job.id]
    assert kw["start_new_session"] and kw["cwd"] == jobs.REPO_ROOT
    assert jobs.get(job.id).pid == 4242
    assert job.id in capsys.readouterr().out


def test_queue_runner_starts_nothing(jobs, launched, monkeypatch, capsys):
    seesaw(monkeypatch, "jobs", "submit", "q", "--runner", "queue")
    assert launched == []
    assert "--poll" in capsys.readouterr().out


def test_auto_runner_prefers_modal_when_configured(monkeypatch):
    from seesaw import jobs_cli

    assert jobs_cli.pick_runner() == "local"
    monkeypatch.setenv("MODAL_RUN_STAGE_URL", "https://example.invalid/start")
    assert jobs_cli.pick_runner() == "modal"
    assert jobs_cli.pick_runner("queue") == "queue"


def test_approve_is_guarded_on_the_gate(jobs, launched, monkeypatch, capsys):
    job = jobs.create("q")
    assert seesaw(monkeypatch, "jobs", "approve", job.id, "--runner", "local") == 1
    assert "no longer awaiting" in capsys.readouterr().err
    assert launched == []

    jobs.advance(job)                                   # scout done, parked before lens
    assert seesaw(monkeypatch, "jobs", "approve", job.id, "--runner", "local") == 0
    assert jobs.get(job.id).status == jobs.QUEUED
    assert len(launched) == 1


def test_retry_only_requeues_a_failed_job(jobs, launched, monkeypatch):
    job = jobs.create("q")
    assert seesaw(monkeypatch, "jobs", "retry", job.id, "--runner", "queue") == 1
    jobs.update(job.id, status=jobs.FAILED, error="lens: boom")
    assert seesaw(monkeypatch, "jobs", "retry", job.id, "--runner", "queue") == 0
    after = jobs.get(job.id)
    assert (after.status, after.error) == (jobs.QUEUED, None)


def test_show_points_at_the_next_step(jobs, monkeypatch, capsys):
    job = jobs.create("q")
    jobs.update(job.id, plan_path="/tmp/plan.md")
    jobs.advance(jobs.get(job.id))
    seesaw(monkeypatch, "jobs", "show", job.id)
    out = capsys.readouterr().out
    assert "awaiting_approval at lens" in out
    assert f"seesaw jobs approve {job.id}" in out and "/tmp/plan.md" in out
    assert seesaw(monkeypatch, "jobs", "show", "nope") == 1


def test_list_and_cancel(jobs, monkeypatch, capsys):
    job = jobs.create("a question worth asking")
    seesaw(monkeypatch, "jobs", "list")
    assert "a question worth asking" in capsys.readouterr().out
    seesaw(monkeypatch, "jobs", "cancel", job.id)
    assert jobs.get(job.id).status == jobs.CANCELLED


def test_follow_streams_the_log_and_stops_when_the_job_does(jobs, monkeypatch, capsys):
    from seesaw import jobs_cli

    job = jobs.create("q")
    jobs.LOGS_DIR.mkdir(parents=True)
    log = jobs.LOGS_DIR / f"{job.id}.log"
    log.write_text("▶ stage: scout\n")

    def tick(_):          # stands in for the worker: more output, then the job finishes
        with open(log, "a") as f:
            f.write("✔ stage complete\n")
        jobs.update(job.id, status=jobs.DONE, stage="done")

    monkeypatch.setattr(jobs_cli.time, "sleep", tick)
    assert jobs_cli.follow(job.id, prompt_gates=False) == 0
    out = capsys.readouterr().out
    assert out.count("▶ stage: scout") == 1 and "✔ stage complete" in out
    assert "■ done at done" in out


def test_follow_reads_the_mirrored_tail_when_the_worker_ran_elsewhere(jobs, monkeypatch, capsys):
    from seesaw import jobs_cli

    job = jobs.create("q")
    jobs.update(job.id, status=jobs.FAILED, log_tail="line one\nTraceback: boom")
    assert jobs_cli.follow(job.id, prompt_gates=False) == 1
    out = capsys.readouterr().out
    assert "Traceback: boom" in out and f"seesaw jobs retry {job.id}" in out


def test_a_lens_stage_with_no_successes_fails_the_job(jobs, monkeypatch):
    """An empty bundle stops at Lens, where retry re-runs it, not at Quill's gate."""
    import types

    from orchestrator.src import worker

    plan = jobs.LOGS_DIR.parent / "plan.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# plan")
    bundle = {"n_total": 2, "n_success": 0, "results": [{"error": "ValueError: got 8 for 3 prompts"}]}
    fake = types.ModuleType("lens_client")
    fake.run_lens = lambda *a, **k: (bundle, plan.parent / "bundle.json")
    monkeypatch.setitem(sys.modules, "orchestrator.src.clients.lens_client", fake)

    job = jobs.create("q")
    jobs.update(job.id, stage="lens", plan_path=str(plan))
    assert worker.run_job(job.id) == 1
    after = jobs.get(job.id)
    assert (after.status, after.stage) == (jobs.FAILED, "lens")
    assert "no experiment succeeded" in after.error and "got 8 for 3 prompts" in after.error
