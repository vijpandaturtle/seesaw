"""Store for research jobs, on SQLite or Postgres.

A *job* is one Scout → Lens → Quill run. Stages execute in a worker
process (orchestrator/src/worker.py), never inside whatever created the
job — a Lens experiment easily outlives the request that asked for it.

This store is the only shared state between the dashboard and the
workers: the dashboard creates jobs and records approvals, workers claim
stages and write back artifact paths.

DATABASE_URL selects Postgres (needed once the dashboard and the workers
are on different machines); without it the store is a local SQLite file,
where WAL handles the cross-process traffic and there's no server to run.
SEESAW_DB_PATH forces SQLite even when DATABASE_URL is set.

    from shared.db import jobs

    job = jobs.create("What heads mediate IOI in GPT-2?", model="gpt2")
    jobs.get(job.id).status        # queued -> running -> awaiting_approval

Something has to pick the job up: `python -m orchestrator.src.worker
--poll`, or the dashboard spawning a worker per job when it runs on the
same host (see seesaw-web).

`stage` always names the stage that is running or up next; `status` says
what is happening to it. A job with stage="lens", status="awaiting_approval"
has finished Scout and is waiting for a human to approve the plan.
"""

from __future__ import annotations

import contextlib
import os
import re
import threading
import signal
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"
LOGS_DIR    = OUTPUTS_DIR / "job_logs"

# Stages in execution order. `stage` is one of these, or "done".
STAGES = ("scout", "lens", "quill")

# Status values.
QUEUED    = "queued"              # stage ready to run, no worker on it yet
RUNNING   = "running"             # a worker is executing `stage`
AWAITING  = "awaiting_approval"   # previous stage done, human gate before `stage`
DONE      = "done"                # all three stages finished
FAILED    = "failed"              # worker raised; see `error`
CANCELLED = "cancelled"           # stopped by the user

ACTIVE_STATUSES = (QUEUED, RUNNING)

_COLUMNS = (
    "question", "model", "status", "stage", "auto_approve",
    "plan_path", "bundle_path", "report_path", "error", "pid", "updated_at",
    "cancel_requested", "worker_id", "log_tail",
)

# Added after the first release; connect() backfills them on open so an
# existing jobs.db keeps working.
_ADDED_COLUMNS = {
    "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
    "worker_id":        "TEXT",
    "log_tail":         "TEXT",
}

# How much worker output to mirror into the row. A frontend on another host
# can't read the log file, so this is what it shows instead.
LOG_TAIL_LINES = 200


# ── Record ───────────────────────────────────────────────────────────────────
@dataclass
class Job:
    id: str
    question: str
    model: str
    status: str
    stage: str
    auto_approve: int
    plan_path: str | None
    bundle_path: str | None
    report_path: str | None
    error: str | None
    pid: int | None
    created_at: float
    updated_at: float
    cancel_requested: int = 0
    worker_id: str | None = None
    log_tail: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(**{k: row[k] for k in row.keys()})

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def completed_stages(self) -> list[str]:
        """Stages that have produced their artifact."""
        if self.stage == "done":
            return list(STAGES)
        return list(STAGES[: STAGES.index(self.stage)])

    @property
    def log_path(self) -> Path:
        return LOGS_DIR / f"{self.id}.log"


# ── Connection ───────────────────────────────────────────────────────────────
def db_path() -> Path:
    """Location of the SQLite file (override with SEESAW_DB_PATH)."""
    return Path(os.environ.get("SEESAW_DB_PATH") or OUTPUTS_DIR / "jobs.db")


def is_postgres() -> bool:
    """Whether the store is backed by Postgres rather than a local file.

    SEESAW_DB_PATH wins when both are set, so pointing at a file is always an
    unambiguous way to get SQLite — tests rely on that, and it means an
    exported DATABASE_URL can't silently redirect a local run at production.
    """
    if os.environ.get("SEESAW_DB_PATH"):
        return False
    return bool(os.environ.get("DATABASE_URL"))


def _table() -> str:
    """How to name the jobs table for the active backend.

    Schema-qualified rather than selected with `SET search_path`, because a
    pooled Postgres endpoint (Neon's `-pooler` host, PgBouncer in transaction
    mode) multiplexes statements across backend connections: session state set
    on one doesn't reliably apply to the next, so search_path silently stops
    holding. Qualifying per statement is unaffected.
    """
    schema = os.environ.get("SEESAW_DB_SCHEMA")
    return f'"{schema}".jobs' if schema and is_postgres() else "jobs"


def _q(sql: str) -> str:
    """Adapt a query written in SQLite style to the active backend."""
    if not is_postgres():
        return sql
    sql = sql.replace("?", "%s")
    table = _table()
    return sql if table == "jobs" else re.sub(r"\bjobs\b", table, sql)


def _ex(conn, sql: str, params: tuple = ()):
    """Execute with backend-appropriate placeholders."""
    return conn.execute(_q(sql), params)


# Column types differ between the two; everything else about the schema is the
# same, so the DDL is generated rather than duplicated.
_BASE_COLUMNS = (
    ("id", "TEXT PRIMARY KEY"),
    ("question", "TEXT NOT NULL"),
    ("model", "TEXT NOT NULL"),
    ("status", "TEXT NOT NULL"),
    ("stage", "TEXT NOT NULL"),
    ("auto_approve", "INTEGER NOT NULL DEFAULT 0"),
    ("plan_path", "TEXT"),
    ("bundle_path", "TEXT"),
    ("report_path", "TEXT"),
    ("error", "TEXT"),
    ("pid", "INTEGER"),
    ("created_at", "{REAL} NOT NULL"),
    ("updated_at", "{REAL} NOT NULL"),
)


def _create_table_sql(real_type: str) -> str:
    cols = ",\n            ".join(
        f"{name:13s} {spec.format(REAL=real_type)}" for name, spec in _BASE_COLUMNS
    )
    return f"CREATE TABLE IF NOT EXISTS jobs (\n            {cols}\n        )"


# One pool per process. Opening a fresh TLS connection to a hosted Postgres
# costs seconds, and the dashboard polls every few seconds, so connections are
# reused rather than made per query.
_pool = None
_pool_key: str | None = None
_pool_lock = threading.Lock()


def _get_pool():
    global _pool, _pool_key

    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    schema = os.environ.get("SEESAW_DB_SCHEMA") or ""
    key = f"{os.environ['DATABASE_URL']}#{schema}"
    if _pool is not None and _pool_key == key:
        return _pool

    # Workers claim jobs from several threads at once, so building the pool
    # has to be serialised: unguarded, two threads each build one and the
    # second closes the pool the first handed out, leaving queries running
    # against a closed connection.
    with _pool_lock:
        if _pool is not None and _pool_key == key:
            return _pool
        previous = _pool

        pool = ConnectionPool(
            os.environ["DATABASE_URL"],
            min_size=1,
            max_size=10,
            # autocommit removes a BEGIN/COMMIT round trip per call. Every
            # operation here is a single statement — including the claim,
            # whose atomicity comes from the UPDATE's WHERE clause rather than
            # from a transaction — so there is nothing multi-statement to
            # protect.
            kwargs={"row_factory": dict_row, "autocommit": True},
            open=True,
        )
        # Schema setup runs once per pool, not once per query — it's four DDL
        # round-trips, which is most of the cost of opening a connection.
        with pool.connection() as conn:
            if schema:
                conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            conn.execute(_q(_create_table_sql("DOUBLE PRECISION")))
            for column, spec in _ADDED_COLUMNS.items():
                conn.execute(
                    _q(f"ALTER TABLE jobs ADD COLUMN IF NOT EXISTS {column} {spec}")
                )

        _pool, _pool_key = pool, key

    if previous is not None:
        previous.close()
    return pool


_sqlite_ready: set[str] = set()


def _connect_sqlite() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    if str(path) not in _sqlite_ready:
        conn.execute(_create_table_sql("REAL"))
        # SQLite has no ADD COLUMN IF NOT EXISTS, so columns are checked first.
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
        for column, spec in _ADDED_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {spec}")
        conn.commit()
        _sqlite_ready.add(str(path))
    return conn


@contextlib.contextmanager
def connect():
    """A connection with the schema in place, as a context manager.

    The only place either driver is named — everything else goes through _ex,
    so switching backends doesn't touch the queries. Postgres connections come
    from a pool and are returned to it on exit; SQLite ones are closed.
    """
    if is_postgres():
        with _get_pool().connection() as conn:
            yield conn
    else:
        conn = _connect_sqlite()
        try:
            with conn:
                yield conn
        finally:
            conn.close()


# ── CRUD ─────────────────────────────────────────────────────────────────────
def create(question: str, model: str = "gpt2", auto_approve: bool = False) -> Job:
    """Insert a new job, queued at the Scout stage.

    Args:
        question: The mech interp question to investigate.
        model: TransformerLens model Lens should target.
        auto_approve: Run all three stages without stopping at the
            human gates between them.

    Returns:
        The created Job.
    """
    now = time.time()
    job = Job(
        id=uuid.uuid4().hex[:8], question=question.strip(), model=model,
        status=QUEUED, stage="scout", auto_approve=int(auto_approve),
        plan_path=None, bundle_path=None, report_path=None, error=None,
        pid=None, created_at=now, updated_at=now,
    )
    with connect() as conn:
        _ex(
            conn,
            "INSERT INTO jobs (id, question, model, status, stage, auto_approve,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (job.id, job.question, job.model, job.status, job.stage,
             job.auto_approve, job.created_at, job.updated_at),
        )
    return job


def get(job_id: str) -> Job | None:
    with connect() as conn:
        row = _ex(conn, "SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return Job.from_row(row) if row else None


def list_jobs(limit: int = 100) -> list[Job]:
    """Newest first."""
    with connect() as conn:
        rows = _ex(
            conn, "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [Job.from_row(r) for r in rows]


def update(job_id: str, **fields) -> None:
    """Write the named columns. Unknown column names raise rather than
    reaching the SQL string."""
    unknown = set(fields) - set(_COLUMNS)
    if unknown:
        raise ValueError(f"unknown job column(s): {sorted(unknown)}")
    fields["updated_at"] = time.time()
    assignments = ", ".join(f"{k} = ?" for k in fields)
    with connect() as conn:
        _ex(conn, f"UPDATE jobs SET {assignments} WHERE id = ?",
            (*fields.values(), job_id))


def delete(job_id: str) -> None:
    """Drop the job record. Artifacts on disk are left alone."""
    with connect() as conn:
        _ex(conn, "DELETE FROM jobs WHERE id = ?", (job_id,))


# ── Stage transitions ────────────────────────────────────────────────────────
def advance(job: Job) -> None:
    """Move a job past the stage it just finished.

    The last stage completes the job; otherwise the next stage is either
    queued (auto_approve) or parked at its human gate.
    """
    idx = STAGES.index(job.stage)
    if idx == len(STAGES) - 1:
        update(job.id, stage="done", status=DONE, pid=None)
    else:
        update(job.id, stage=STAGES[idx + 1],
               status=QUEUED if job.auto_approve else AWAITING)


def approve(job_id: str) -> None:
    """Clear the human gate so a worker can pick up the pending stage."""
    update(job_id, status=QUEUED, error=None)


def claim(job_id: str, worker_id: str) -> Job | None:
    """Take ownership of a queued job, atomically.

    The UPDATE only matches while the row is still queued, so when several
    workers poll at once exactly one sees a row change and the rest get None.
    This is what replaces spawn_worker once the process that creates jobs is
    no longer the process that runs them.

    Returns:
        The claimed Job, or None if another worker got there first.
    """
    with connect() as conn:
        cur = _ex(
            conn,
            "UPDATE jobs SET status = ?, worker_id = ?, error = NULL, updated_at = ?"
            " WHERE id = ? AND status = ?",
            (RUNNING, worker_id, time.time(), job_id, QUEUED),
        )
        if cur.rowcount == 0:
            return None
    return get(job_id)


def claim_next(worker_id: str) -> Job | None:
    """Claim the oldest queued job, or None if there is nothing to run."""
    with connect() as conn:
        rows = _ex(
            conn, "SELECT id FROM jobs WHERE status = ? ORDER BY created_at", (QUEUED,)
        ).fetchall()
    for row in rows:
        job = claim(row["id"], worker_id)
        if job is not None:
            return job
    return None


def next_queued_id() -> str | None:
    """Oldest queued job id, without taking it.

    For pollers: claiming here and then calling run_job would double-claim,
    and run_job's own claim — the one that makes concurrent workers safe —
    only matches rows still queued. Peek, then let run_job do the claiming.
    """
    with connect() as conn:
        row = _ex(
            conn,
            "SELECT id FROM jobs WHERE status = ? ORDER BY created_at LIMIT 1",
            (QUEUED,),
        ).fetchone()
    return row["id"] if row else None


def request_cancel(job_id: str) -> None:
    """Ask a job to stop.

    Sets a flag the worker checks between stages, because the worker may be
    on a different host than whoever clicked cancel. A local worker is also
    signalled directly so an idle job stops immediately.
    """
    job = get(job_id)
    update(job_id, cancel_requested=1)
    if job and job.pid:
        try:
            os.kill(job.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass          # gone, not ours, or on another machine entirely
    if job and not job.is_active:
        update(job_id, status=CANCELLED, pid=None)


def cancel_requested(job_id: str) -> bool:
    """Whether someone asked this job to stop (read by the worker)."""
    job = get(job_id)
    return bool(job and job.cancel_requested)


def cancel(job_id: str) -> None:
    """Stop a job now and mark it cancelled."""
    request_cancel(job_id)
    update(job_id, status=CANCELLED, pid=None)


# ── Worker output ────────────────────────────────────────────────────────────
def tail_log(job_id: str, n_lines: int = 60) -> str:
    """Recent worker output.

    Prefers the log file, which only exists on the machine that ran the job.
    A frontend deployed away from the workers falls back to the copy the
    worker mirrored into the row.
    """
    path = LOGS_DIR / f"{job_id}.log"
    if path.exists():
        return "\n".join(path.read_text(errors="replace").splitlines()[-n_lines:])
    job = get(job_id)
    if job and job.log_tail:
        return "\n".join(job.log_tail.splitlines()[-n_lines:])
    return ""


def sync_log(job_id: str, n_lines: int = LOG_TAIL_LINES) -> None:
    """Mirror the tail of the local log file into the job row.

    Called by the worker at stage boundaries so anyone reading the store from
    another host sees progress. Cheap enough to call often; the row holds
    only the last n_lines.
    """
    path = LOGS_DIR / f"{job_id}.log"
    if not path.exists():
        return
    tail = "\n".join(path.read_text(errors="replace").splitlines()[-n_lines:])
    update(job_id, log_tail=tail)
