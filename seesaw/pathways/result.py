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

    `verdict` is one word from the pathway's own fixed vocabulary (each
    pathway's module docstring lists it). `evidence` is the handful of
    sentences, each carrying its number, that the verdict rests on — what a
    reader who won't open the plots needs. `stopped_early` says why the
    pathway stopped before its last stage, if it did; "no effect to explain"
    is a result, not a failure.
    """

    pathway: str
    steps: dict[str, InterpResult] = field(default_factory=dict)
    findings: dict = field(default_factory=dict)
    verdict: str | None = None
    evidence: list[str] = field(default_factory=list)
    stopped_early: str | None = None

    @property
    def plot_paths(self) -> list[Path]:
        return [p for step in self.steps.values() for p in step.plot_paths]

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
        lines += ["## Steps run", ""] + [f"1. `{name}`" for name in self.steps]
        if self.plot_paths:
            lines += ["", "## Plots", ""] + [f"- {p}" for p in self.plot_paths]
        return "\n".join(lines) + "\n"
