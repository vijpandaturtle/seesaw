"""Run the eval suite and persist each result.

    python scripts/run_eval_suite.py                    # every task
    python scripts/run_eval_suite.py --task quill_weak_bundle_1

Every task runs one agent (Scout, Lens, or Quill) alone: minutes and cents
each. Scores are judged afterwards by check_eval_baselines.py.

A task that raises is reported and the run continues: one broken task
shouldn't hide the results of the others.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
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
    else:
        selected = list(tasks.values())

    selected.sort(key=lambda t: t.id)
    print(f"running {len(selected)} task(s)\n")

    failures = 0
    for task in selected:
        print(f"═══ {task.id} (target={task.target})")
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
