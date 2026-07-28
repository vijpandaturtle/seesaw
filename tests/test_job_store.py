"""Behavioural spec for the job store.

Written against SQLite, but deliberately free of SQLite specifics: these are
the guarantees the store has to provide whichever database is behind it. When
the store moves to Postgres, this suite is what says the port is correct.

The properties that matter are the ones the rest of the system leans on:

  * a queued job is claimed by exactly one worker, however many are competing
  * approving advances a job without a worker having to poll for permission
  * cancelling is visible to a worker that may be on another machine
  * new columns can be added to an existing database without losing rows

Run with:  pytest tests/ -q
"""

from __future__ import annotations

import concurrent.futures
import importlib
import os
import sqlite3
import uuid

import pytest


def _postgres_url() -> str | None:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ.get("DATABASE_URL")


@pytest.fixture(params=["sqlite", "postgres"])
def jobs(request, tmp_path, monkeypatch):
    """A fresh, isolated store per test — once per backend.

    Postgres runs in a throwaway schema so the same assertions can be made
    against a real database without touching the live jobs table. It's skipped
    when DATABASE_URL isn't configured.
    """
    if request.param == "postgres":
        url = _postgres_url()
        if not url:
            pytest.skip("DATABASE_URL not set — skipping the Postgres backend")
        schema = f"test_{uuid.uuid4().hex[:12]}"
        monkeypatch.delenv("SEESAW_DB_PATH", raising=False)
        monkeypatch.setenv("DATABASE_URL", url)
        monkeypatch.setenv("SEESAW_DB_SCHEMA", schema)
    else:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("SEESAW_DB_SCHEMA", raising=False)
        monkeypatch.setenv("SEESAW_DB_PATH", str(tmp_path / "jobs.db"))

    from shared.db import jobs as module

    importlib.reload(module)
    yield module

    if request.param == "postgres":
        with module.connect() as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            conn.commit()


# ── lifecycle ────────────────────────────────────────────────────────────────
def test_new_job_is_queued_at_the_first_stage(jobs):
    job = jobs.create("what heads mediate IOI?", model="gpt2")
    assert job.status == jobs.QUEUED
    assert job.stage == "scout"
    assert job.completed_stages == []
    assert jobs.get(job.id).question == "what heads mediate IOI?"


def test_advance_parks_at_the_gate_then_completes(jobs):
    job = jobs.create("q")
    jobs.advance(job)                       # scout done
    job = jobs.get(job.id)
    assert (job.stage, job.status) == ("lens", jobs.AWAITING)
    assert job.completed_stages == ["scout"]

    jobs.approve(job.id)
    assert jobs.get(job.id).status == jobs.QUEUED

    jobs.advance(jobs.get(job.id))          # lens done
    jobs.approve(job.id)
    jobs.advance(jobs.get(job.id))          # quill done
    job = jobs.get(job.id)
    assert (job.stage, job.status) == ("done", jobs.DONE)
    assert job.completed_stages == ["scout", "lens", "quill"]


def test_auto_approve_skips_the_gates(jobs):
    job = jobs.create("q", auto_approve=True)
    jobs.advance(job)
    after = jobs.get(job.id)
    assert after.status == jobs.QUEUED, "auto-approve should not stop at a gate"
    assert after.stage == "lens"


# ── claiming ─────────────────────────────────────────────────────────────────
def test_claim_takes_a_queued_job(jobs):
    job = jobs.create("q")
    claimed = jobs.claim(job.id, "worker-1")
    assert claimed is not None
    assert claimed.status == jobs.RUNNING
    assert claimed.worker_id == "worker-1"


def test_claim_refuses_a_job_that_is_not_queued(jobs):
    job = jobs.create("q")
    assert jobs.claim(job.id, "worker-1") is not None
    assert jobs.claim(job.id, "worker-2") is None, "a running job was claimed twice"


def test_next_queued_id_does_not_take_the_job(jobs):
    """The poller peeks and lets run_job do the atomic claim. If peeking
    claimed, the claim inside run_job would then find nothing queued — which
    is exactly the bug that left jobs stuck in `running` with no worker."""
    job = jobs.create("q")
    assert jobs.next_queued_id() == job.id
    assert jobs.get(job.id).status == jobs.QUEUED
    assert jobs.claim(job.id, "worker-1") is not None


def test_concurrent_claims_are_exclusive(jobs):
    """Twenty jobs, eight workers, every job claimed exactly once."""
    ids = {jobs.create(f"q{i}").id for i in range(20)}

    def drain(n: int) -> list[str]:
        got = []
        while True:
            job = jobs.claim_next(f"w{n}")
            if job is None:
                return got
            got.append(job.id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        claimed = [i for batch in pool.map(drain, range(8)) for i in batch]

    assert len(claimed) == len(set(claimed)), "a job was claimed by two workers"
    assert set(claimed) == ids, "a queued job was never claimed"


# ── cancellation ─────────────────────────────────────────────────────────────
def test_cancel_is_visible_to_a_worker_elsewhere(jobs):
    """Cancelling can't rely on signalling a pid — the worker may be on
    another machine, so it has to be a flag the worker reads."""
    job = jobs.create("q")
    jobs.claim(job.id, "worker-1")
    jobs.request_cancel(job.id)
    assert jobs.cancel_requested(job.id) is True


def test_cancel_marks_the_job_cancelled(jobs):
    job = jobs.create("q")
    jobs.cancel(job.id)
    assert jobs.get(job.id).status == jobs.CANCELLED


# ── updates and integrity ────────────────────────────────────────────────────
def test_update_rejects_unknown_columns(jobs):
    """Field names reach an f-string in the UPDATE, so they're validated
    rather than trusted."""
    job = jobs.create("q")
    with pytest.raises(ValueError):
        jobs.update(job.id, bogus_column="x")


def test_artifact_paths_round_trip(jobs):
    job = jobs.create("q")
    jobs.update(job.id, plan_path="/tmp/plan.md", bundle_path="/tmp/b.json")
    stored = jobs.get(job.id)
    assert stored.plan_path == "/tmp/plan.md"
    assert stored.bundle_path == "/tmp/b.json"


def test_delete_removes_only_that_job(jobs):
    keep = jobs.create("keep")
    drop = jobs.create("drop")
    jobs.delete(drop.id)
    assert jobs.get(drop.id) is None
    assert jobs.get(keep.id) is not None


def test_list_is_newest_first(jobs):
    first = jobs.create("first")
    second = jobs.create("second")
    listed = [j.id for j in jobs.list_jobs()]
    assert listed.index(second.id) < listed.index(first.id)


def test_missing_job_returns_none(jobs):
    assert jobs.get("deadbeef") is None


# ── migration ────────────────────────────────────────────────────────────────
def test_columns_are_added_to_an_existing_database(tmp_path, monkeypatch):
    # SQLite-specific: exercises the pre-migration on-disk schema.
    """A store created before the multi-host columns existed must keep its
    rows when opened by the current code."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE jobs (
             id TEXT PRIMARY KEY, question TEXT NOT NULL, model TEXT NOT NULL,
             status TEXT NOT NULL, stage TEXT NOT NULL,
             auto_approve INTEGER NOT NULL DEFAULT 0,
             plan_path TEXT, bundle_path TEXT, report_path TEXT,
             error TEXT, pid INTEGER,
             created_at REAL NOT NULL, updated_at REAL NOT NULL)"""
    )
    conn.execute(
        "INSERT INTO jobs VALUES ('old1','legacy question','gpt2','done','done',"
        "0,NULL,NULL,NULL,NULL,NULL,1.0,1.0)"
    )
    conn.commit()
    conn.close()

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SEESAW_DB_PATH", str(db))
    from shared.db import jobs as module

    importlib.reload(module)

    survivor = module.get("old1")
    assert survivor is not None, "migration dropped an existing row"
    assert survivor.question == "legacy question"
    assert survivor.cancel_requested == 0
    assert survivor.worker_id is None
