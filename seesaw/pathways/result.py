"""The result shape every `seesaw.pathways` pathway returns."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from seesaw.interp.result import InterpResult
from shared.evidence import assess

# Steps named after something other than the tool they ran.
_STEP_TOOLS = {"ablation_sweep": "ablation", "steering": "steering_vector"}

# Called as each step's result is recorded: (pathway, step, result, seconds
# since the previous step was recorded). The job worker listens to put
# pathway steps on a job's timeline as they happen.
StepListener = Callable[[str, str, InterpResult, float], None]
_listeners: list[StepListener] = []


@contextlib.contextmanager
def listening(listener: StepListener):
    """Call `listener` for every pathway step recorded inside the block."""
    _listeners.append(listener)
    try:
        yield
    finally:
        _listeners.remove(listener)


class _Steps(dict):
    """A step dict that tells listeners about each step as it's recorded.

    Pathways run their steps in order and record each one as it finishes, so
    the time since the previous one is (close to) how long the step took.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathway = ""
        self._last = time.perf_counter()

    def __setitem__(self, name, result):
        super().__setitem__(name, result)
        now = time.perf_counter()
        seconds, self._last = now - self._last, now
        for listener in list(_listeners):
            try:
                listener(self.pathway, name, result, seconds)
            except Exception:                         # noqa: BLE001 — a listener can't stop a pathway
                pass


@dataclass
class PathwayResult:
    """Every step's raw InterpResult, plus the conclusions the pathway drew.

    `steps` keeps each intermediate result by step name, in run order, so a
    reader (or a grader) can audit how the pathway reached its `findings`.
    `findings` is the pathway's own deterministic verdict: no LLM involved.

    `verdict` is one word from the pathway's own fixed vocabulary (each
    pathway's module docstring lists it). `evidence` is the handful of
    sentences, each carrying its number, that the verdict rests on — what a
    reader who won't open the plots needs. `stopped_early` says why the
    pathway stopped before its last stage, if it did; "no effect to explain"
    is a result, not a failure.
    """

    pathway: str
    steps: dict[str, InterpResult] = field(default_factory=_Steps)
    findings: dict = field(default_factory=dict)
    verdict: str | None = None
    evidence: list[str] = field(default_factory=list)
    stopped_early: str | None = None

    def __post_init__(self):
        if not isinstance(self.steps, _Steps):
            self.steps = _Steps(self.steps)
        self.steps.pathway = self.pathway

    @property
    def plot_paths(self) -> list[Path]:
        return [p for step in self.steps.values() for p in step.plot_paths]

    def step_evidence(self) -> dict[str, dict]:
        """shared.evidence.assess for every step, keyed by step name."""
        return {name: assess(_STEP_TOOLS.get(name, name), getattr(step, "data", None) or {},
                             backend="seesaw")
                for name, step in self.steps.items()}

    def stop(self, verdict: str, reason: str) -> "PathwayResult":
        self.verdict, self.stopped_early = verdict, reason
        self.evidence.append(f"Stopped: {reason}.")
        return self

    def to_markdown(self) -> str:
        """A one-page summary: verdict, the evidence behind it, then the details."""
        lines = [f"# {self.pathway}", "", f"**Verdict:** `{self.verdict}`", ""]
        if self.evidence:
            lines += ["## Evidence", ""] + [f"- {e}" for e in self.evidence] + [""]
        if self.findings:
            lines += ["## Findings", ""]
            for key, value in self.findings.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")
        evidence = self.step_evidence()
        lines += ["## Steps run", ""] + [f"1. `{name}` — {evidence[name]['label']}" for name in self.steps]
        if self.plot_paths:
            lines += ["", "## Plots", ""] + [f"- {p}" for p in self.plot_paths]
        return "\n".join(lines) + "\n"
