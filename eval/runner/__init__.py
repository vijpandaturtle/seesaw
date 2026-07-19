"""Task runner: execute eval/tasks/*.yaml against the live agents and grade.

    from eval.runner import run_task
    from eval.tasks import load_all
    result = run_task(load_all()["quill_weak_bundle_1"])

CLI:
    python -m eval.runner <task_id> [--no-persist]
    python -m eval.runner --list
"""

from .core import AgentRun, RUNS_DIR, TaskResult, run_task

__all__ = ["run_task", "TaskResult", "AgentRun", "RUNS_DIR"]
