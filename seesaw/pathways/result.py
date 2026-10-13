"""The result shape every `seesaw.pathways` pathway returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from seesaw.interp.result import InterpResult


@dataclass
class PathwayResult:
    """Every step's raw InterpResult, plus the conclusions the pathway drew.

    `steps` keeps each intermediate result by step name, in run order, so a
    reader (or a grader) can audit how the pathway reached its `findings`.
    `findings` is the pathway's own deterministic verdict: no LLM involved.
    """

    pathway: str
    steps: dict[str, InterpResult] = field(default_factory=dict)
    findings: dict = field(default_factory=dict)

    @property
    def plot_paths(self) -> list[Path]:
        return [p for step in self.steps.values() for p in step.plot_paths]
