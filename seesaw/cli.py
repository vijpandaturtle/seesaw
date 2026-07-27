"""The `seesaw` command — thin wrapper over the library API.

    seesaw plan "<question>"              Scout only: question -> research plan
    seesaw run "<question>" [--skip-hitl] Full Scout -> Lens -> Quill pipeline
    seesaw critique <bundle.json>         Quill only: bundle -> critique report
    seesaw eval <task_id> | --list        Run an eval task against the live agents
    seesaw fixtures [--llm [AGENT]]       Grader meta-eval (offline free; --llm costs)

The dashboard is a separate project — see github.com/vijpandaturtle/seesaw-web.

Every subcommand is a thin call into `import seesaw` / `seesaw.evals` — anything
the CLI does, a script or notebook can do too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


def _cmd_plan(args) -> int:
    import seesaw

    path = seesaw.run_scout(args.question)
    print(f"\nplan: {path}")
    return 0


def _cmd_run(args) -> int:
    import seesaw

    seesaw.run_pipeline(args.question, skip_hitl=args.skip_hitl)
    return 0


def _cmd_critique(args) -> int:
    import seesaw

    report, path = seesaw.run_quill(Path(args.bundle))
    print(f"\nassessment: {report.overall_assessment}\nreport: {path}")
    return 0


def _cmd_eval(args) -> int:
    from seesaw import evals

    tasks = evals.load_tasks()
    if args.list or not args.task_id:
        for tid, t in sorted(tasks.items()):
            print(f"{tid:30s} target={t.target or '-':9s} {len(t.graders)} graders")
        return 0
    if args.task_id not in tasks:
        print(f"unknown task {args.task_id!r}; try `seesaw eval --list`", file=sys.stderr)
        return 1
    result = evals.run_task(tasks[args.task_id])
    for r in result.grader_results:
        mark = "⚠️ " if r.score is None else ("✅" if r.score >= 0.5 else "❌")
        print(f"{mark} {r.criterion:45s} score={r.score}")
    print(f"metrics: {result.metrics}")
    return 0


def _cmd_fixtures(args) -> int:
    from seesaw import evals

    if args.llm is not None:
        agent = args.llm or None
        summary = evals.check_llm(agent=agent)
        for r in summary["results"]:
            mark = "✅" if r["ok"] else "❌"
            print(f"{mark} {r['id']:42s} expected={r['expected']} actual={r['actual']}")
        print(f"judge agreement: {summary['matched']}/{summary['ran']}")
        return 0 if summary["matched"] == summary["ran"] else 1
    res = evals.check_offline()
    print(f"{res['ran_offline']}/{res['total']} fixtures ran offline; "
          f"{len(res['mismatches'])} mismatches")
    for m in res["mismatches"]:
        print(f"  ❌ {m}")
    return 0 if not res["mismatches"] else 1


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="seesaw", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="Scout: research question -> plan")
    p.add_argument("question")
    p.set_defaults(fn=_cmd_plan)

    p = sub.add_parser("run", help="full Scout -> Lens -> Quill pipeline")
    p.add_argument("question")
    p.add_argument("--skip-hitl", action="store_true", help="no approval checkpoints")
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("critique", help="Quill: results bundle -> critique")
    p.add_argument("bundle", help="path to results_bundle.json")
    p.set_defaults(fn=_cmd_critique)

    p = sub.add_parser("eval", help="run an eval task against the live agents")
    p.add_argument("task_id", nargs="?")
    p.add_argument("--list", action="store_true")
    p.set_defaults(fn=_cmd_eval)

    p = sub.add_parser("fixtures", help="grader meta-eval")
    p.add_argument("--llm", nargs="?", const="", metavar="AGENT",
                   help="run LLM fixtures (optionally one agent) — costs judge calls")
    p.set_defaults(fn=_cmd_fixtures)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
