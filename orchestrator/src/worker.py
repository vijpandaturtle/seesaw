"""Run a research job's stages outside the UI process.

    python -m orchestrator.src.worker <job_id>

Streamlit reruns its script on every interaction and tears everything
down when the browser disconnects, so pipeline stages can't live there —
a Lens experiment outlives the page that launched it. This worker takes
a job created by the UI, runs stages in order, and writes each artifact
path back to the job store as it goes.

With auto_approve off (the default), the worker runs one stage and exits,
leaving the job parked at the next human gate; the dashboard's approve
button queues the stage and spawns a fresh worker. With auto_approve on,
a single worker runs all three stages back to back.

Everything printed here (and by the agents themselves) lands in the job's
log file — see shared/db/jobs.spawn_worker.
"""

from __future__ import annotations

import sys
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
def run_job(job_id: str) -> int:
    """Run queued stages for this job until it blocks, finishes, or fails.

    Returns:
        Process exit code — 0 unless a stage raised.
    """
    while True:
        job = jobs.get(job_id)
        if job is None:
            print(f"job {job_id} not found", file=sys.stderr)
            return 1
        if job.status != jobs.QUEUED:
            # Nothing to claim: awaiting approval, cancelled, done, or another
            # worker already took it.
            print(f"job {job_id} is {job.status} at stage {job.stage} — nothing to run")
            return 0

        jobs.update(job.id, status=jobs.RUNNING, error=None)
        print(f"\n▶ stage: {job.stage}")

        try:
            STAGE_FNS[job.stage](job)
        except Exception as exc:                     # noqa: BLE001 — reported to the UI
            traceback.print_exc()
            jobs.update(job.id, status=jobs.FAILED,
                        error=f"{job.stage}: {type(exc).__name__}: {exc}"[:1000],
                        pid=None)
            return 1

        # Re-read: the stage wrote artifact paths, and the user may have
        # cancelled while it ran.
        job = jobs.get(job_id)
        if job is None or job.status == jobs.CANCELLED:
            print("job cancelled during stage")
            return 0

        jobs.advance(job)
        after = jobs.get(job_id)
        print(f"✔ stage complete — now {after.status} at {after.stage}")

        if after.status != jobs.QUEUED:
            jobs.update(job_id, pid=None)
            return 0


def main() -> int:
    load_dotenv()
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    return run_job(sys.argv[1])


if __name__ == "__main__":
    raise SystemExit(main())
