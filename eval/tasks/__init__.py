"""Declarative task specs (eval/tasks/*.yaml) + their loader.

    from eval.tasks import load_all, load_task, Task
    tasks = load_all()          # {task_id: Task}, validated

See loader.py for the task schema and the grader-type vocabulary.
"""

from .loader import (
    GRADER_TYPE_SPECS,
    METRIC_TYPE_SPECS,
    GraderSpec,
    Task,
    TaskValidationError,
    TrackedMetric,
    load_all,
    load_task,
)

__all__ = [
    "load_all", "load_task", "Task", "GraderSpec", "TrackedMetric",
    "TaskValidationError", "GRADER_TYPE_SPECS", "METRIC_TYPE_SPECS",
]
