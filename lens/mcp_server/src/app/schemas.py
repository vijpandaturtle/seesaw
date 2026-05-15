from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ExperimentSpec:
    """Describes a single experiment for Lens to run."""
    name: str
    tool: str
    model_name: str
    prompts: list[str]
    what_to_measure: str
    hypothesis_tested: str
    expected_outcome: str
    tool_kwargs: dict = field(default_factory=dict)   # extra tool-specific params (e.g. io_tokens)


@dataclass
class ExperimentResult:
    """Output from a single Lens tool run."""
    name: str
    tool: str
    model_name: str
    prompts: list[str]
    summary: str = ""
    plot_paths: list[Path] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    status: Literal["success", "failed", "timeout"] = "success"
    error: str | None = None


@dataclass
class ExperimentBundle:
    """Full output from a Lens run — passed to Quill for critique."""
    research_question: str
    model_name: str
    results: list[ExperimentResult]

    @property
    def successful(self) -> list[ExperimentResult]:
        return [r for r in self.results if r.status == "success"]

    @property
    def failed(self) -> list[ExperimentResult]:
        return [r for r in self.results if r.status != "success"]
