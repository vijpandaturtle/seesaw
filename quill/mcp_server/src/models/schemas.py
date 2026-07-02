import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


# ── Lens output schemas (mirrors lens/mcp_server/src/app/schemas.py) ──────────
@dataclass
class ExperimentResult:
    name: str
    tool: str
    model_name: str
    prompts: list[str]
    summary: str = ""
    plot_paths: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    status: Literal["success", "failed", "timeout"] = "success"
    error: str | None = None


@dataclass
class ExperimentBundle:
    research_question: str
    model_name: str
    results: list[ExperimentResult]

    @property
    def successful(self) -> list[ExperimentResult]:
        return [r for r in self.results if r.status == "success"]

    @property
    def failed(self) -> list[ExperimentResult]:
        return [r for r in self.results if r.status != "success"]

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentBundle":
        with open(path) as f:
            raw = json.load(f)
        results = [ExperimentResult(**r) for r in raw["results"]]
        return cls(
            research_question=raw["research_question"],
            model_name=raw["model_name"],
            results=results,
        )


# ── Quill structured output models ────────────────────────────────────────────
class ExperimentCritique(BaseModel):
    """Per-experiment assessment."""
    experiment_name: str
    validity: Literal["strong", "moderate", "weak"]
    conclusions_supported: bool
    issues: list[str]                    # confounds, methodology problems
    alternative_explanations: list[str]  # other ways to read the results
    positive_aspects: list[str]          # what the experiment does well


class ResearchGap(BaseModel):
    """An experiment that should have been run but wasn't."""
    description: str
    severity: Literal["critical", "important", "minor"]
    why_needed: str        # what question it would answer
    suggested_tool: str    # which Lens tool would address it


class FollowUpSpec(BaseModel):
    """Concrete experiment spec that Lens can execute directly."""
    name: str
    tool: str              # logit_lens | attention_pattern | ablation | activation_patching | direct_logit_attribution
    rationale: str         # why this follow-up is needed
    what_to_measure: str
    hypothesis: str
    priority: Literal["high", "medium", "low"]


# ── Intermediate structured outputs for each node ─────────────────────────────
class ExperimentCritiquesOutput(BaseModel):
    critiques: list[ExperimentCritique]
    overall_assessment: Literal["strong", "moderate", "weak"]
    overall_summary: str


class GapsOutput(BaseModel):
    gaps: list[ResearchGap]
    coverage_verdict: str  # 1-2 sentences on how well the experiments cover the question


class FollowUpsOutput(BaseModel):
    followups: list[FollowUpSpec]


# ── Final report ───────────────────────────────────────────────────────────────
@dataclass
class CritiqueReport:
    research_question: str
    model_name: str
    overall_assessment: Literal["strong", "moderate", "weak"]
    overall_summary: str
    coverage_verdict: str
    critiques: list[ExperimentCritique]
    gaps: list[ResearchGap]
    followups: list[FollowUpSpec]
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "research_question":  self.research_question,
            "model_name":         self.model_name,
            "overall_assessment": self.overall_assessment,
            "overall_summary":    self.overall_summary,
            "coverage_verdict":   self.coverage_verdict,
            "generated_at":       self.generated_at,
            "critiques":          [c.model_dump() for c in self.critiques],
            "gaps":               [g.model_dump() for g in self.gaps],
            "followups":          [f.model_dump() for f in self.followups],
        }
