"""Run a research job's stages.

    python -m orchestrator.src.worker <job_id>   one job, then exit
    python -m orchestrator.src.worker --poll     claim queued jobs forever

A Lens experiment outlives the request that asked for it, so stages run
here rather than in whatever created the job. The worker takes a job from
the store, runs stages in order, and writes each artifact path back as it
goes.

With auto_approve off (the default), it runs one stage and exits, leaving
the job parked at the next human gate; approving queues the stage again
for a worker to claim. With auto_approve on, one worker runs all three
stages back to back.

--poll is the entry point when nothing starts workers on demand — the
dashboard deployed away from the agents, for instance. Several pollers
can run at once; claims are atomic, so no two take the same job.

Everything printed here (and by the agents) goes to the job's log file
under outputs/job_logs/, mirrored into the row by jobs.sync_log so a
remote dashboard can read it.
"""

from __future__ import annotations

import os
import socket
import sys
import time
import traceback

from dotenv import load_dotenv

from shared.db import jobs
from shared.db.jobs import Job

# Lens honours the plan's stated model, so the UI's choice is appended as
# an explicit override rather than hoping Scout wrote the right one.
_MODEL_OVERRIDE = (
    "\n\n**Target Model (required, overrides any other model mentioned above)**: {model}\n"
)


# ── Stages ───────────────────────────────────────────────────────────────────
def run_scout_stage(job: Job) -> None:
    from .clients.scout_client import run_scout

    plan_path = run_scout(job.question, thread_id=f"job-{job.id}-scout")
    jobs.update(job.id, plan_path=str(plan_path))


def run_lens_stage(job: Job) -> None:
    from .clients.lens_client import run_lens

    if not job.plan_path:
        raise RuntimeError("no research plan on this job — Scout stage did not save one")

    from pathlib import Path

    plan_text = Path(job.plan_path).read_text()
    bundle, bundle_path = run_lens(
        plan_text + _MODEL_OVERRIDE.format(model=job.model),
        thread_id=f"job-{job.id}-lens",
    )
    jobs.update(job.id, bundle_path=str(bundle_path))
    print(f"   {bundle['n_success']}/{bundle['n_total']} experiments succeeded")


def run_quill_stage(job: Job) -> None:
    from pathlib import Path

    from .clients.quill_client import run_quill

    if not job.bundle_path:
        raise RuntimeError("no results bundle on this job — Lens stage produced none")

    report, report_path = run_quill(Path(job.bundle_path), thread_id=f"job-{job.id}-quill")
    jobs.update(job.id, report_path=str(report_path))
    print(f"   assessment: {report.overall_assessment}")


STAGE_FNS = {
    "scout": run_scout_stage,
    "lens":  run_lens_stage,
    "quill": run_quill_stage,
}


# ── Driver ───────────────────────────────────────────────────────────────────
def run_job(job_id: str, worker_id: str | None = None) -> int:
    """Run queued stages for this job until it blocks, finishes, or fails.

    Returns:
        Process exit code — 0 unless a stage raised.
    """
    worker_id = worker_id or f"pid-{os.getpid()}"
    while True:
        job = jobs.claim(job_id, worker_id)
        if job is None:
            current = jobs.get(job_id)
            if current is None:
                print(f"job {job_id} not found", file=sys.stderr)
                return 1
            # Awaiting approval, cancelled, done, or another worker got it.
            print(f"job {job_id} is {current.status} at stage {current.stage} — nothing to run")
            return 0

        if jobs.cancel_requested(job_id):
            print("job cancelled before stage started")
            jobs.update(job_id, status=jobs.CANCELLED, pid=None)
            return 0

        print(f"\n▶ stage: {job.stage}")
        try:
            STAGE_FNS[job.stage](job)
        except Exception as exc:                     # noqa: BLE001 — reported to the UI
            traceback.print_exc()
            jobs.update(job.id, status=jobs.FAILED,
                        error=f"{job.stage}: {type(exc).__name__}: {exc}"[:1000],
                        pid=None)
            jobs.sync_log(job_id)
            return 1

        # Re-read: the stage wrote artifact paths, and the user may have
        # asked to cancel while it ran.
        job = jobs.get(job_id)
        if job is None or job.status == jobs.CANCELLED or job.cancel_requested:
            print("job cancelled during stage")
            jobs.update(job_id, status=jobs.CANCELLED, pid=None)
            jobs.sync_log(job_id)
            return 0

        jobs.advance(job)
        after = jobs.get(job_id)
        print(f"✔ stage complete — now {after.status} at {after.stage}")
        jobs.sync_log(job_id)

        if after.status != jobs.QUEUED:
            jobs.update(job_id, pid=None)
            return 0


def poll(worker_id: str | None = None, interval: float = 5.0) -> int:
    """Claim and run queued jobs forever.

    This is the backend's entry point once the frontend is deployed somewhere
    that can't spawn processes: Streamlit inserts a queued row, this loop
    picks it up. Safe to run several of these — claims are atomic, so two
    pollers never take the same job.
    """
    worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
    print(f"👷 poller {worker_id} started — checking every {interval}s")
    while True:
        # Peek rather than claim: run_job does the atomic claim, and that's
        # what decides the winner when several pollers see the same row.
        job_id = jobs.next_queued_id()
        if job_id is None:
            time.sleep(interval)
            continue
        job = jobs.get(job_id)
        print(f"\n📥 picking up job {job_id} at stage {job.stage}: {job.question[:60]}")
        try:
            run_job(job_id, worker_id=worker_id)
        except Exception:                            # noqa: BLE001 — keep polling
            traceback.print_exc()


def main() -> int:
    load_dotenv()
    args = sys.argv[1:]
    if args == ["--poll"]:
        return poll()
    if len(args) != 1:
        print("usage: python -m orchestrator.src.worker <job_id> | --poll",
              file=sys.stderr)
        return 2
    return run_job(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
