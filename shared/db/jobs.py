"""SQLite-backed store for research jobs.

A *job* is one Scout → Lens → Quill run. Stages execute in a worker
process (orchestrator/src/worker.py), not inside Streamlit — a Lens
experiment easily outlives the page that started it, and Streamlit drops
everything when the browser disconnects.

This store is the only shared state between the UI and the workers: the
UI creates jobs and records approvals, workers claim stages and write
back artifact paths. WAL mode handles the cross-process traffic, so
there's no server to run.

    from shared.db import jobs

    job = jobs.create("What heads mediate IOI in GPT-2?", model="gpt2")
    jobs.spawn_worker(job.id)
    jobs.get(job.id).status        # running -> awaiting_approval -> ...

`stage` always names the stage that is running or up next; `status` says
what is happening to it. A job with stage="lens", status="awaiting_approval"
has finished Scout and is waiting for a human to approve the plan.
"""

from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
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
)


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


def connect() -> sqlite3.Connection:
    """Open a connection, creating the database and schema if needed."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id           TEXT PRIMARY KEY,
            question     TEXT NOT NULL,
            model        TEXT NOT NULL,
            status       TEXT NOT NULL,
            stage        TEXT NOT NULL,
            auto_approve INTEGER NOT NULL DEFAULT 0,
            plan_path    TEXT,
            bundle_path  TEXT,
            report_path  TEXT,
            error        TEXT,
            pid          INTEGER,
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


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
        conn.execute(
            "INSERT INTO jobs (id, question, model, status, stage, auto_approve,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (job.id, job.question, job.model, job.status, job.stage,
             job.auto_approve, job.created_at, job.updated_at),
        )
    return job


def get(job_id: str) -> Job | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return Job.from_row(row) if row else None


def list_jobs(limit: int = 100) -> list[Job]:
    """Newest first."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
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
        conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?",
                     (*fields.values(), job_id))


def delete(job_id: str) -> None:
    """Drop the job record. Artifacts on disk are left alone."""
    with connect() as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


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


def cancel(job_id: str) -> None:
    """Mark a job cancelled and stop its worker if one is running."""
    job = get(job_id)
    if job and job.pid:
        try:
            os.kill(job.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass          # already gone, or owned by another user
    update(job_id, status=CANCELLED, pid=None)


# ── Worker process ───────────────────────────────────────────────────────────
def spawn_worker(job_id: str) -> int:
    """Start a detached worker for this job and return its pid.

    stdout/stderr go to the job's log file, which is how the dashboard
    shows agent progress — the agents print as they work.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = (LOGS_DIR / f"{job_id}.log").open("a", buffering=1)
    log.write(f"\n{'═' * 70}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] worker starting\n")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "orchestrator.src.worker", job_id],
        cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,     # survives the Streamlit process
    )
    update(job_id, pid=proc.pid)
    return proc.pid


def tail_log(job_id: str, n_lines: int = 60) -> str:
    path = LOGS_DIR / f"{job_id}.log"
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-n_lines:])
