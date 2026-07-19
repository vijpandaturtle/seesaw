"""Run eval tasks as LangSmith experiments.

Flow:
  1. ensure_dataset() — one LangSmith example per task (inputs = {"task_id"}).
  2. evaluate(target=..., evaluators=[unpack_grader_results]) — target executes
     the real agents via run_task(); the single evaluator fans the TaskResult's
     grader results out into per-criterion feedback scores ({"results": [...]}),
     so each criterion shows as its own column in the experiment UI.
  3. LANGSMITH_TRACING is enabled for the duration, so every Claude/tool call the
     agents make is traced and nested under its example row.

Usage:
    python -m eval.runner.langsmith_runner quill_weak_bundle_1 quill_strong_bundle_1
    python -m eval.runner.langsmith_runner --all-cheap     # scout/quill/lens targets only

View results: https://smith.langchain.com → Datasets & Experiments → seesaw-tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from ..tasks import Task, load_all
from .core import run_task

DATASET_NAME = "seesaw-tasks"


def _jsonable(obj) -> dict:
    """Round-trip through json to strip Paths/enums/dataclasses for upload."""
    return json.loads(json.dumps(obj, default=str))


def ensure_dataset(tasks: dict[str, Task]):
    from langsmith import Client

    client = Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        existing = {e.inputs.get("task_id") for e in client.list_examples(dataset_id=dataset.id)}
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Seesaw live-agent eval tasks (eval/tasks/*.yaml)",
        )
        existing = set()

    new = [t for tid, t in tasks.items() if tid not in existing]
    if new:
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[{"task_id": t.id} for t in new],
            outputs=[{"desc": t.desc, "target": t.target} for t in new],
        )
        print(f"dataset {DATASET_NAME!r}: added {len(new)} examples ({len(existing)} existed)")
    return dataset


def target(inputs: dict) -> dict:
    """LangSmith target: execute the task's agents + graders, return the result."""
    task = load_all()[inputs["task_id"]]
    result = run_task(task, persist=True)
    d = result.to_dict()
    d.pop("trajectory", None)   # traced anyway; keep the row payload light
    return _jsonable(d)


def unpack_grader_results(run, example) -> dict:
    """One evaluator that fans out every grader result as its own feedback key."""
    out = []
    for g in (run.outputs or {}).get("grader_results", []):
        comment = g.get("detail") or ""
        if g.get("error"):
            comment = f"[error: {g['error']}] {comment}"
        if g.get("needs_human"):
            comment = f"[needs human] {comment}"
        out.append({"key": g["criterion"], "score": g.get("score"), "comment": comment[:500]})
    scores = [g.get("score") for g in (run.outputs or {}).get("grader_results", [])
              if g.get("score") is not None]
    out.append({"key": "task_mean", "score": sum(scores) / len(scores) if scores else None,
                "comment": f"mean of {len(scores)} scored criteria"})
    return {"results": out}


def run_experiment(task_ids: list[str], *, prefix: str = "seesaw") -> None:
    from langsmith.evaluation import evaluate

    os.environ.setdefault("LANGSMITH_TRACING", "true")   # nest agent traces per row
    tasks = load_all()
    unknown = [t for t in task_ids if t not in tasks]
    if unknown:
        raise SystemExit(f"unknown task ids: {unknown}")

    selected = {tid: tasks[tid] for tid in task_ids}
    dataset = ensure_dataset(tasks)

    from langsmith import Client
    client = Client()
    example_ids = [e.id for e in client.list_examples(dataset_id=dataset.id)
                   if e.inputs.get("task_id") in selected]

    print(f"running {len(example_ids)} task(s) as experiment '{prefix}-…' — this "
          f"invokes the real agents and costs API credits.")
    evaluate(
        target,
        data=client.list_examples(dataset_id=dataset.id, example_ids=example_ids),
        evaluators=[unpack_grader_results],
        experiment_prefix=prefix,
        max_concurrency=1,          # agents share output dirs; keep runs serial
    )


def main() -> int:
    load_dotenv()
    if not os.getenv("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY not set — set it in .env first.", file=sys.stderr)
        return 1
    parser = argparse.ArgumentParser(description="Run eval tasks as LangSmith experiments")
    parser.add_argument("task_ids", nargs="*", help="task ids to run")
    parser.add_argument("--all-cheap", action="store_true",
                        help="run every non-pipeline task (no full Scout->Lens->Quill chains)")
    parser.add_argument("--prefix", default="seesaw")
    args = parser.parse_args()

    ids = args.task_ids
    if args.all_cheap:
        ids = [tid for tid, t in load_all().items() if t.target != "pipeline"]
    if not ids:
        parser.error("give task ids or --all-cheap")
    run_experiment(ids, prefix=args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
