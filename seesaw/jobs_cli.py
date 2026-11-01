"""`seesaw jobs …` — submit and drive jobs in the shared job store from a terminal.

    seesaw jobs submit "<question>" [--model M] [--auto-approve] [--follow]
    seesaw jobs list [--limit N] [--json]
    seesaw jobs show <id> [--json]
    seesaw jobs approve <id> [--follow]
    seesaw jobs retry <id> [--follow]
    seesaw jobs cancel <id>
    seesaw jobs logs <id> [-n N] [--follow]
    seesaw jobs open <id>

Unlike `seesaw run`, which runs the pipeline in this process and records
nothing, these go through the same job store the dashboard reads: a job
submitted here shows up there, and one queued there can be approved here.

Who executes a stage is the runner, chosen like the dashboard chooses it:

    modal   MODAL_RUN_STAGE_URL is set: ask the deployed Modal function
    local   a detached worker process on this machine (the default otherwise)
    queue   start nothing; a `python -m orchestrator.src.worker --poll` claims it

`--runner` overrides the choice. A local worker writes its output to the
job's log file and outlives this command, so Ctrl-C on --follow stops
watching, not the job.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from dataclasses import asdict

RUNNERS = ("auto", "local", "modal", "queue")
FOLLOW_INTERVAL = 2.0
DEFAULT_WEB_URL = "http://localhost:3000"


def _jobs():
    from shared.db import jobs

    return jobs


# ── Runners ──────────────────────────────────────────────────────────────────
def pick_runner(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "modal" if os.environ.get("MODAL_RUN_STAGE_URL") else "local"


def start_stage(job_id: str, runner: str = "auto") -> str:
    """Hand the job's queued stage to a runner. Returns the runner used, or
    "queue" when nothing was started (by choice, or because the runner failed:
    the row stays queued either way, so a poller or a retry can still take it)."""
    runner = pick_runner(runner)
    if runner == "modal":
        return "modal" if _start_modal(job_id) else "queue"
    if runner == "local":
        return "local" if _start_local(job_id) else "queue"
    return "queue"


def _start_modal(job_id: str) -> bool:
    body = json.dumps({"job_id": job_id, "key": os.environ.get("MODAL_INVOKE_TOKEN", "")}).encode()
    req = urllib.request.Request(os.environ["MODAL_RUN_STAGE_URL"], data=body,
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return 200 <= r.status < 300
    except OSError as exc:
        print(f"modal runner unreachable ({exc}); job left queued", file=sys.stderr)
        return False


def _start_local(job_id: str) -> bool:
    """A detached worker, launched the way the dashboard's local mode launches one,
    so the log tab reads the same whichever started it."""
    jobs = _jobs()
    jobs.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(jobs.LOGS_DIR / f"{job_id}.log", "a") as log:
        try:
            child = subprocess.Popen(
                [sys.executable, "-u", "-m", "orchestrator.src.worker", job_id],
                cwd=jobs.REPO_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True,      # survives this command and its Ctrl-C
            )
        except OSError as exc:
            print(f"could not start a worker ({exc}); job left queued", file=sys.stderr)
            return False
    jobs.update(job_id, pid=child.pid)   # lets `jobs cancel` signal it directly
    return True


# ── Display ──────────────────────────────────────────────────────────────────
def _age(ts: float) -> str:
    s = max(0, int(time.time() - ts))
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if s >= size:
            return f"{s // size}{unit}"
    return f"{s}s"


def _web_url(job_id: str) -> str:
    return f"{os.environ.get('SEESAW_WEB_URL', DEFAULT_WEB_URL).rstrip('/')}/jobs/{job_id}"


def _describe(job) -> str:
    lines = [
        f"job      {job.id}",
        f"question {job.question}",
        f"model    {job.model}",
        f"status   {job.status} at {job.stage}" + ("  (auto-approve)" if job.auto_approve else ""),
        f"updated  {_age(job.updated_at)} ago",
    ]
    for label, path in (("plan", job.plan_path), ("bundle", job.bundle_path), ("report", job.report_path)):
        if path:
            lines.append(f"{label:8s} {path}")
    if job.error:
        lines.append(f"error    {job.error}")
    lines.append(f"web      {_web_url(job.id)}")
    hint = _next_step(job)
    if hint:
        lines.append(f"\nnext: {hint}")
    return "\n".join(lines)


def _next_step(job) -> str | None:
    jobs = _jobs()
    if job.status == jobs.AWAITING:
        what = {"lens": f"the plan ({job.plan_path})", "quill": f"the results ({job.bundle_path})"}.get(job.stage, "it")
        return f"review {what}, then `seesaw jobs approve {job.id}`"
    if job.status == jobs.FAILED:
        return f"`seesaw jobs logs {job.id}`, then `seesaw jobs retry {job.id}`"
    if job.status == jobs.QUEUED:
        return "waiting for a worker; nothing claims it unless a runner was started or a poller is running"
    return None


def _get(job_id: str):
    job = _jobs().get(job_id)
    if job is None:
        print(f"no job {job_id!r}; `seesaw jobs list` shows recent ones", file=sys.stderr)
    return job


# ── Following ────────────────────────────────────────────────────────────────
def follow(job_id: str, runner: str = "auto", prompt_gates: bool = True) -> int:
    """Stream the job's log until it stops moving. At an approval gate, ask on a
    terminal (or print how to approve later); return 1 if it ends failed."""
    jobs = _jobs()
    printed = _LogCursor(job_id)
    try:
        while True:
            printed.flush()
            job = jobs.get(job_id)
            if job is None:
                print(f"job {job_id} was deleted", file=sys.stderr)
                return 1
            if job.is_active:
                time.sleep(FOLLOW_INTERVAL)
                continue
            printed.flush()
            print(f"\n■ {job.status} at {job.stage}")
            if job.status == jobs.AWAITING and prompt_gates and sys.stdin.isatty():
                if _confirm(f"{_next_step(job)}\napprove now? [y/N] "):
                    _approve(job_id, runner)
                    continue
            elif _next_step(job):
                print(f"next: {_next_step(job)}")
            return 1 if job.status == jobs.FAILED else 0
    except KeyboardInterrupt:
        print(f"\nstopped following; the job carries on. `seesaw jobs show {job_id}`")
        return 130


def _confirm(question: str) -> bool:
    try:
        return input(question).strip().lower() in ("y", "yes")
    except EOFError:
        return False


class _LogCursor:
    """Prints only output not shown yet: by byte offset in the local log file, or,
    when the worker ran elsewhere, by matching the end of the mirrored tail."""

    def __init__(self, job_id: str):
        self.job_id, self.offset, self.seen = job_id, 0, ""

    def flush(self) -> None:
        jobs = _jobs()
        path = jobs.LOGS_DIR / f"{self.job_id}.log"
        if path.exists():
            with open(path, errors="replace") as f:
                f.seek(self.offset)
                text = f.read()
                self.offset = f.tell()
            if text:
                print(text, end="", flush=True)
            return
        job = jobs.get(self.job_id)
        tail = (job.log_tail or "") if job else ""
        if tail == self.seen:
            return
        new = tail[len(self.seen):] if tail.startswith(self.seen) else tail
        if new:
            print(new if new.endswith("\n") else new + "\n", end="", flush=True)
        self.seen = tail


# ── Commands ─────────────────────────────────────────────────────────────────
def _approve(job_id: str, runner: str) -> bool:
    if not _jobs().approve_if_awaiting(job_id):
        print(f"job {job_id} is no longer awaiting approval", file=sys.stderr)
        return False
    used = start_stage(job_id, runner)
    print(f"approved; stage handed to the {used} runner" if used != "queue" else "approved; left queued")
    return True


def cmd_submit(args) -> int:
    job = _jobs().create(args.question, model=args.model, auto_approve=args.auto_approve)
    used = start_stage(job.id, args.runner)
    print(f"job {job.id} queued at scout; runner: {used}\nweb: {_web_url(job.id)}")
    if used == "queue":
        print("nothing will run it until a poller claims it: python -m orchestrator.src.worker --poll")
    return follow(job.id, args.runner) if args.follow else 0


def cmd_list(args) -> int:
    rows = _jobs().list_jobs(limit=args.limit)
    if args.json:
        print(json.dumps([asdict(j) | {"log_tail": None} for j in rows], indent=2, default=str))
        return 0
    if not rows:
        print("no jobs yet; `seesaw jobs submit \"<question>\"` starts one")
        return 0
    print(f"{'ID':8s}  {'STATUS':17s}  {'STAGE':5s}  {'AGE':>4s}  QUESTION")
    for j in rows:
        q = j.question if len(j.question) <= 60 else j.question[:57] + "..."
        print(f"{j.id:8s}  {j.status:17s}  {j.stage:5s}  {_age(j.created_at):>4s}  {q}")
    return 0


def cmd_show(args) -> int:
    job = _get(args.job_id)
    if job is None:
        return 1
    print(json.dumps(asdict(job), indent=2, default=str) if args.json else _describe(job))
    return 0


def cmd_approve(args) -> int:
    if _get(args.job_id) is None or not _approve(args.job_id, args.runner):
        return 1
    return follow(args.job_id, args.runner) if args.follow else 0


def cmd_retry(args) -> int:
    if _get(args.job_id) is None:
        return 1
    if not _jobs().retry_if_failed(args.job_id):
        print(f"job {args.job_id} is not in a failed state", file=sys.stderr)
        return 1
    used = start_stage(args.job_id, args.runner)
    print(f"re-queued; runner: {used}")
    return follow(args.job_id, args.runner) if args.follow else 0


def cmd_cancel(args) -> int:
    if _get(args.job_id) is None:
        return 1
    _jobs().cancel(args.job_id)
    print(f"job {args.job_id} cancelled")
    return 0


def cmd_logs(args) -> int:
    if _get(args.job_id) is None:
        return 1
    if args.follow:
        return follow(args.job_id, prompt_gates=False)
    print(_jobs().tail_log(args.job_id, n_lines=args.lines) or "(no output yet)")
    return 0


def cmd_open(args) -> int:
    url = _web_url(args.job_id)
    print(url)
    webbrowser.open(url)
    return 0


def add_parser(sub) -> None:
    """Register `jobs` and its subcommands on the top-level `seesaw` parser."""
    import argparse

    jobs_p = sub.add_parser("jobs", help="submit and drive jobs in the shared job store",
                            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    js = jobs_p.add_subparsers(dest="jobs_command", required=True)

    def runner_arg(p):
        p.add_argument("--runner", choices=RUNNERS, default="auto",
                       help="who executes the stage (default: modal if MODAL_RUN_STAGE_URL is set, else local)")

    def follow_arg(p):
        p.add_argument("-f", "--follow", action="store_true",
                       help="stream the log until the job stops, asking at approval gates")

    p = js.add_parser("submit", help="create a job and start its first stage")
    p.add_argument("question")
    p.add_argument("--model", default="gpt2", help="model Lens should target (default: gpt2)")
    p.add_argument("--auto-approve", action="store_true", help="run all stages without approval gates")
    runner_arg(p)
    follow_arg(p)
    p.set_defaults(fn=cmd_submit)

    p = js.add_parser("list", help="recent jobs, newest first")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_list)

    p = js.add_parser("show", help="one job's status, artifacts and next step")
    p.add_argument("job_id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_show)

    p = js.add_parser("approve", help="clear the approval gate and run the next stage")
    p.add_argument("job_id")
    runner_arg(p)
    follow_arg(p)
    p.set_defaults(fn=cmd_approve)

    p = js.add_parser("retry", help="re-run a failed stage")
    p.add_argument("job_id")
    runner_arg(p)
    follow_arg(p)
    p.set_defaults(fn=cmd_retry)

    p = js.add_parser("cancel", help="stop a job")
    p.add_argument("job_id")
    p.set_defaults(fn=cmd_cancel)

    p = js.add_parser("logs", help="worker output")
    p.add_argument("job_id")
    p.add_argument("-n", "--lines", type=int, default=60)
    p.add_argument("-f", "--follow", action="store_true", help="keep streaming until the job stops")
    p.set_defaults(fn=cmd_logs)

    p = js.add_parser("open", help="open the job in the dashboard (SEESAW_WEB_URL)")
    p.add_argument("job_id")
    p.set_defaults(fn=cmd_open)
