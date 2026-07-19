"""CLI: python -m eval.runner <task_id> | --list"""

import argparse
import sys

from dotenv import load_dotenv

from ..tasks import load_all
from . import run_task


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run one eval task against the live agents")
    parser.add_argument("task_id", nargs="?", help="task id from eval/tasks/*.yaml")
    parser.add_argument("--list", action="store_true", help="list available tasks")
    parser.add_argument("--no-persist", action="store_true", help="don't write outputs/eval_runs/")
    args = parser.parse_args()

    tasks = load_all()
    if args.list or not args.task_id:
        for tid, t in sorted(tasks.items()):
            print(f"{tid:30s} target={t.target or '-':9s} {len(t.graders)} graders")
        return 0
    if args.task_id not in tasks:
        print(f"unknown task {args.task_id!r}; use --list", file=sys.stderr)
        return 1

    result = run_task(tasks[args.task_id], persist=not args.no_persist)
    print(f"\n═══ {result.task_id} ═══")
    for r in result.grader_results:
        mark = "✅" if (r.score or 0) >= 0.5 else ("⚠️ " if r.score is None else "❌")
        print(f"{mark} {r.criterion:45s} score={r.score} {('— ' + r.detail[:100]) if r.detail else ''}")
        if r.error:
            print(f"     error: {r.error[:150]}")
    print(f"\nmetrics: {result.metrics}")
    if result.run.errors:
        print(f"run errors: {result.run.errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
