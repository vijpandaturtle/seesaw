"""Facade over the eval suite (eval/): graders, fixtures, tasks, runner.

    from seesaw import evals

    evals.check_offline()                       # grader meta-eval (free, offline)
    evals.check_llm(agent="scout")              # judge calibration (costs API calls)
    tasks = evals.load_tasks()                  # {task_id: Task}
    result = evals.run_task(tasks["quill_weak_bundle_1"])
    result.scores

See DOCS.md → Evaluation for the architecture (fixtures test the graders; tasks
test the agents).
"""

from __future__ import annotations

__all__ = ["check_offline", "check_llm", "load_tasks", "load_fixtures",
           "run_task", "REGISTRY"]


def __getattr__(name: str):
    import importlib

    if name in ("check_offline", "check_llm"):
        return getattr(importlib.import_module("eval.fixtures"), name)
    if name == "load_fixtures":
        return importlib.import_module("eval.fixtures").load_all
    if name == "load_tasks":
        return importlib.import_module("eval.tasks").load_all
    if name == "run_task":
        return importlib.import_module("eval.runner").run_task
    if name == "REGISTRY":
        return importlib.import_module("eval.graders").REGISTRY
    raise AttributeError(f"module 'seesaw.evals' has no attribute {name!r}")
