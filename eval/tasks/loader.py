"""Parse and validate declarative task specs, and map grader `type`s to backing.

A task is a YAML file describing a unit of work for the agents plus how to grade
the result. Shape (one `task:` mapping per file):

    task:
      id: "ioi_circuit_1"
      desc: "..."
      # optional Seesaw fields, passed through in Task.extra:
      question: "..."          # research question fed to the pipeline
      model: gpt2              # target model
      target: pipeline         # scout | lens | quill | pipeline
      graders:
        - type: <grader_type>  # see GRADER_TYPE_SPECS
          <type-specific config...>
      tracked_metrics:
        - type: <metric_type>  # see METRIC_TYPE_SPECS
          metrics: [...]

Grader `type`s are a small, extensible vocabulary. Each type is a WAY OF CHECKING;
the runner (execution layer, added later) dispatches on `type` and, where the
check is Seesaw-domain judgment, calls into eval/graders/. Mapping:

  deterministic_tests -> run named pytest node(s); the Lens numeric layer lives here
                         (backs lens.numeric_correctness / robustness via assertions)
  llm_rubric          -> eval.graders.judge() with the rubric file as the prompt
                         (backs scout.hypothesis_specificity, lens.interpretation_*, ...)
  static_analysis     -> run linters (ruff/mypy/bandit) over produced code artifacts
  state_check         -> assert on produced artifacts/state (plan saved, bundle schema)
  tool_calls          -> assert the agent's trajectory used the required tool calls
                         (backs scout.tools_called_properly / lens.plan_execution_fidelity)

tracked_metrics are recorded, not scored — operational telemetry (backs
scout.efficiency / lens.operational_cost).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TASKS_DIR = Path(__file__).resolve().parent

# Grader type -> required / optional config keys (everything except `type`).
GRADER_TYPE_SPECS: dict[str, dict[str, set[str]]] = {
    "deterministic_tests": {"required": {"required"}, "optional": set()},
    "llm_rubric":          {"required": {"rubric"},   "optional": {"model", "target"}},
    "static_analysis":     {"required": {"commands"}, "optional": {"paths"}},
    "state_check":         {"required": {"expect"},   "optional": {"target"}},
    "tool_calls":          {"required": {"required"}, "optional": {"target", "forbidden"}},
}

# Metric type -> allowed metric names. Extend as new telemetry is captured.
METRIC_TYPE_SPECS: dict[str, set[str]] = {
    "transcript": {"n_turns", "n_toolcalls", "n_total_tokens", "n_input_tokens", "n_output_tokens"},
    "latency": {"time_to_first_token", "output_tokens_per_sec", "time_to_last_token", "wall_time"},
}


@dataclass
class GraderSpec:
    type: str
    config: dict[str, Any]           # everything on the grader entry except `type`


@dataclass
class TrackedMetric:
    type: str
    metrics: list[str]


@dataclass
class Task:
    id: str
    desc: str
    graders: list[GraderSpec]
    tracked_metrics: list[TrackedMetric]
    file_path: Path
    extra: dict[str, Any] = field(default_factory=dict)   # question, model, target, ...

    # Convenience accessors for the common Seesaw fields.
    @property
    def question(self) -> str | None:
        return self.extra.get("question")

    @property
    def target(self) -> str | None:
        return self.extra.get("target")


class TaskValidationError(ValueError):
    """Raised with the offending file + a specific reason."""


def _validate_grader(spec: GraderSpec, where: str) -> None:
    rules = GRADER_TYPE_SPECS.get(spec.type)
    if rules is None:
        raise TaskValidationError(
            f"{where}: unknown grader type {spec.type!r}. "
            f"Known types: {sorted(GRADER_TYPE_SPECS)}"
        )
    missing = rules["required"] - spec.config.keys()
    if missing:
        raise TaskValidationError(
            f"{where}: grader {spec.type!r} missing required key(s) {sorted(missing)}"
        )
    unknown = spec.config.keys() - rules["required"] - rules["optional"]
    if unknown:
        raise TaskValidationError(
            f"{where}: grader {spec.type!r} has unexpected key(s) {sorted(unknown)}; "
            f"allowed: {sorted(rules['required'] | rules['optional'])}"
        )


def _validate_metric(m: TrackedMetric, where: str) -> None:
    allowed = METRIC_TYPE_SPECS.get(m.type)
    if allowed is None:
        raise TaskValidationError(
            f"{where}: unknown tracked_metric type {m.type!r}. Known: {sorted(METRIC_TYPE_SPECS)}"
        )
    bad = [x for x in m.metrics if x not in allowed]
    if bad:
        raise TaskValidationError(
            f"{where}: metric type {m.type!r} has unknown metric(s) {bad}; allowed: {sorted(allowed)}"
        )


def _parse(raw: dict, path: Path) -> Task:
    if "task" not in raw:
        raise TaskValidationError(f"{path.name}: top-level `task:` key missing")
    body = raw["task"]
    for req in ("id", "desc"):
        if req not in body:
            raise TaskValidationError(f"{path.name}: task missing required field {req!r}")

    graders, metrics, extra = [], [], {}
    for k, v in body.items():
        if k in ("id", "desc", "graders", "tracked_metrics"):
            continue
        extra[k] = v

    for i, g in enumerate(body.get("graders", []) or []):
        if "type" not in g:
            raise TaskValidationError(f"{path.name}: graders[{i}] missing `type`")
        graders.append(GraderSpec(type=g["type"], config={k: v for k, v in g.items() if k != "type"}))

    for i, m in enumerate(body.get("tracked_metrics", []) or []):
        if "type" not in m:
            raise TaskValidationError(f"{path.name}: tracked_metrics[{i}] missing `type`")
        metrics.append(TrackedMetric(type=m["type"], metrics=list(m.get("metrics", []) or [])))

    task = Task(id=body["id"], desc=body["desc"], graders=graders,
                tracked_metrics=metrics, file_path=path, extra=extra)

    for i, spec in enumerate(task.graders):
        _validate_grader(spec, f"{path.name} graders[{i}]")
    for i, m in enumerate(task.tracked_metrics):
        _validate_metric(m, f"{path.name} tracked_metrics[{i}]")
    return task


def load_task(path: str | Path) -> Task:
    path = Path(path)
    return _parse(yaml.safe_load(path.read_text()) or {}, path)


def load_all(directory: str | Path = TASKS_DIR) -> dict[str, Task]:
    """Load every *.yaml/*.yml task in `directory`, keyed by task id."""
    tasks: dict[str, Task] = {}
    for path in sorted(Path(directory).glob("*.y*ml")):
        task = load_task(path)
        if task.id in tasks:
            raise TaskValidationError(f"duplicate task id {task.id!r} ({path.name})")
        tasks[task.id] = task
    return tasks
