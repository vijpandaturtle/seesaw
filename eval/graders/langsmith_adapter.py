"""Bridge the pure graders into LangSmith's `evaluate()` harness.

The graders in this package are framework-free: `(artifact, ...) -> GraderResult`.
LangSmith's `evaluate()` wants evaluator callables shaped `(run, example) -> dict`
(the shape eval_scout.py / eval_quill.py already return). This module converts
between the two so a grader can be dropped into a LangSmith experiment unchanged.

Two pieces:
  - to_langsmith_dict(GraderResult)  -> the {"key","score","comment"} dict LangSmith
                                        records. Dependency-free (no langsmith import).
  - evaluator(grader, extract)       -> a (run, example) callable for `evaluate(...,
                                        evaluators=[...])`. `extract` maps the LangSmith
                                        run/example onto the grader's arguments; that
                                        mapping is task-specific and lives with the
                                        dataset definition, not here.

Usage (in a task/eval script, later):

    from langsmith.evaluation import evaluate
    from eval.graders import lens
    from eval.graders.langsmith_adapter import evaluator

    ioi = evaluator(
        lens.numeric_correctness,
        extract=lambda run, example: {
            "result": run.outputs["result"],
            "expected_heads": example.outputs["expected_top_heads"],
            "baseline_range": tuple(example.outputs["baseline_range"]),
        },
    )
    evaluate(target, data="seesaw-lens-ioi", evaluators=[ioi])
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .base import GraderResult


def _compact(metrics: dict, limit: int = 400) -> str:
    try:
        s = json.dumps(metrics, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(metrics)
    return s if len(s) <= limit else s[:limit] + "…"


def to_langsmith_dict(res: GraderResult, *, key: str | None = None) -> dict[str, Any]:
    """Convert a GraderResult to the dict LangSmith's evaluate() records.

    - key defaults to "<agent>.<criterion>" so scores group per agent in the UI.
    - score is passed through (float, or None for needs_human / not-applicable —
      LangSmith renders a None-score result without averaging it into numeric stats).
    - comment carries the one-line detail plus the raw metrics, so a failing score
      stays inspectable in the trace without opening the source fixture.
    """
    parts = []
    if res.needs_human:
        parts.append("[needs human]")
    if res.error:
        parts.append(f"[grader error: {res.error}]")
    if res.detail:
        parts.append(res.detail)
    if res.metrics:
        parts.append(f"metrics={_compact(res.metrics)}")
    return {
        "key": key or f"{res.agent}.{res.criterion}",
        "score": res.score,
        "comment": " | ".join(parts),
    }


def evaluator(
    grader: Callable[..., GraderResult],
    extract: Callable[[Any, Any], dict],
    *,
    key: str | None = None,
    name: str | None = None,
) -> Callable[[Any, Any], dict[str, Any]]:
    """Wrap a grader into a LangSmith evaluator `(run, example) -> dict`.

    `extract(run, example) -> kwargs` maps the experiment's run outputs and the
    dataset example onto the grader's parameters. A grader that raises (including a
    failed LLM-judge call) is caught and reported as a None-score result rather than
    aborting the whole experiment, so one bad example can't sink the batch.
    """
    def _evaluator(run, example) -> dict[str, Any]:
        try:
            res = grader(**extract(run, example))
        except Exception as exc:  # noqa: BLE001 — one example must not kill the batch
            from .base import GraderType
            res = GraderResult(
                criterion=getattr(grader, "__name__", "grader"), agent="?",
                grader_type=GraderType.CODE, score=None,
                error=repr(exc), detail="grader raised",
            )
        return to_langsmith_dict(res, key=key)

    _evaluator.__name__ = name or key or getattr(grader, "__name__", "grader")
    return _evaluator
