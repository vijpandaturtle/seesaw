"""Run a subset of the eval suite and persist each result.

    python scripts/run_eval_suite.py --suite cheap

Suites exist because cost varies by two orders of magnitude:

    cheap     single-agent tasks — Scout, Quill, or Lens alone. Minutes and
              cents each, which is what CI should normally run.
    pipeline  the full Scout -> Lens -> Quill tasks. Roughly an hour and a few
              dollars apiece, so these are for a release check, not a merge.
    all       everything.

A task that raises is reported and the run continues: one broken task
shouldn't hide the results of the other fourteen.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["cheap", "pipeline", "all"], default="cheap")
    parser.add_argument("--task", action="append", help="run specific task ids instead")
    args = parser.parse_args()

    from seesaw import evals

    tasks = evals.load_tasks()

    if args.task:
        selected = [tasks[t] for t in args.task if t in tasks]
        unknown = [t for t in args.task if t not in tasks]
        if unknown:
            print(f"unknown task(s): {unknown}", file=sys.stderr)
            return 1
    elif args.suite == "cheap":
        selected = [t for t in tasks.values() if (t.target or "pipeline") != "pipeline"]
    elif args.suite == "pipeline":
        selected = [t for t in tasks.values() if (t.target or "pipeline") == "pipeline"]
    else:
        selected = list(tasks.values())

    selected.sort(key=lambda t: t.id)
    print(f"running {len(selected)} task(s) from suite {args.suite!r}\n")

    failures = 0
    for task in selected:
        print(f"═══ {task.id} (target={task.target or 'pipeline'})")
        try:
            result = evals.run_task(task)
        except Exception:                       # noqa: BLE001 — keep going
            failures += 1
            traceback.print_exc()
            print(f"  {task.id} raised; continuing\n")
            continue

        for grader in result.grader_results:
            score = grader.score
            mark = "·" if score is None else ("✓" if score >= 0.5 else "✗")
            print(f"  {mark} {grader.criterion:50s} {score}")
        print()

    if failures:
        print(f"{failures} task(s) raised", file=sys.stderr)
    # Scores are judged by check_eval_baselines.py; a raised task is an
    # infrastructure problem and fails here.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
