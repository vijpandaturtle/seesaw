"""Shared types every grader returns, plus the LLM-judge helper.

A grader is a pure function: it takes an already-produced artifact (a Scout plan,
a Lens ExperimentResult, a Quill CritiqueReport) plus whatever reference it needs,
and returns a GraderResult. Graders do NOT run the agents and do NOT know where
their inputs come from — wiring them to datasets/fixtures ("tasks") is a separate
layer added later. This keeps each grader independently testable and reusable.

Three grader kinds (see eval/EVAL_MATRIX):
  - CODE  : a deterministic rule, no model call.
  - LLM   : a model-as-judge over free text where there's no single right answer.
  - HUMAN : packages what a human reviewer needs to see; score is left None and
            needs_human=True. These never call anything — they format a payload.

Every grader accepts artifacts as either the real dataclass (ExperimentResult,
CritiqueReport, ...) or a plain dict (e.g. deserialized JSON), via the `field()`
accessor below — so graders stay decoupled from the agent packages and can run
against serialized fixtures without importing torch/langgraph.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any


class GraderType(str, Enum):
    CODE = "code"
    LLM = "llm"
    HUMAN = "human"


@dataclass
class GraderResult:
    """One grader's verdict on one artifact.

    score is normalized to [0, 1] where higher is better, or None when the grader
    can't produce a number (needs a human, or not applicable to this artifact).
    metrics carries the raw numbers behind the score — the "Metrics to measure"
    column of the matrix — so a failing score is always inspectable.
    """
    criterion: str                       # matrix row, e.g. "hypothesis_specificity"
    agent: str                           # "scout" | "lens" | "quill"
    grader_type: GraderType
    score: float | None                  # [0,1], higher better; None if n/a or human
    metrics: dict[str, Any] = dc_field(default_factory=dict)
    detail: str = ""                     # one-line human-readable explanation
    needs_human: bool = False            # True for HUMAN graders / deferred judgment
    error: str | None = None             # set if the grader itself failed to run

    @property
    def passed(self) -> bool | None:
        """Convenience boolean at a 0.5 threshold; None when score is None."""
        return None if self.score is None else self.score >= 0.5


def field(obj: Any, name: str, default: Any = None) -> Any:
    """Read `name` off a dataclass/pydantic object OR a dict, uniformly.

    Lets every grader accept either the live artifact or a JSON-deserialized dict
    without caring which — the reason graders don't import the agent schemas.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ── LLM judge ────────────────────────────────────────────────────────────────
# Default judge model. Opus 4.8 is the current most-capable model; note it REJECTS
# temperature/top_p/top_k (400), so the judge is constructed without any sampling
# param — do not add temperature here. Override per-call via `model=` if you want a
# cheaper judge (e.g. "claude-haiku-4-5") for high-volume grading.
JUDGE_MODEL = "claude-opus-4-8"


def judge(prompt: str, schema: type, *, model: str = JUDGE_MODEL):
    """Run a structured LLM-as-judge call, returning an instance of `schema`.

    `schema` is a pydantic BaseModel describing the fields the judge must return.
    langchain-anthropic is imported lazily so that importing a grader module (and
    running the CODE graders) never requires the LLM stack to be installed.
    """
    from langchain_anthropic import ChatAnthropic  # lazy: CODE graders don't need it

    # No temperature: Opus 4.8 rejects sampling params. Determinism for the judge
    # comes from the model + a tightly-scoped rubric, not temperature=0.
    llm = ChatAnthropic(model=model, max_tokens=1024)
    return llm.with_structured_output(schema).invoke(prompt)
