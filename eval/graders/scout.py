"""Scout graders — grade a Research Plan (freeform markdown) and its trajectory.

Scout emits markdown via save_research_plan(plan: str), NOT a structured object,
so the CODE graders here work over plan text + light structure heuristics rather
than a schema. Each function maps to one row of the Scout section of EVAL_MATRIX.

Inputs a grader may take:
  plan        : the plan markdown string
  trajectory  : list of {"name": str, "args": dict} tool calls (from the run trace)
  retrieved   : the search results Scout actually saw (for groundedness)
  reference   : ground-truth facts for the question (expected model, paper, tools)

All are supplied by the caller (the not-yet-built task layer) — graders don't run
Scout.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request

from .base import GraderResult, GraderType, judge

AGENT = "scout"

# Canonical Scout tool names (mirrors scout/mcp_server/src/tools/). Kept as a flat
# constant so this module needn't import the tool package.
SCOUT_TOOLS = {"search_arxiv", "search_arxiv_web", "search_web", "scrape_url", "save_research_plan"}


# ── Code graders ─────────────────────────────────────────────────────────────
def plan_well_formed(plan: str) -> GraderResult:
    """Does the plan contain the required structural pieces (hypotheses, a target
    model, and at least one concrete experiment)? Precondition gate — a failure
    here is usually a save/format bug, not a reasoning one."""
    text = (plan or "").lower()
    checks = {
        "has_hypothesis": bool(re.search(r"hypothesis|hypotheses", text)),
        "has_model": bool(re.search(r"\bmodel\b|gpt-?2|pythia|llama", text)),
        "has_experiment": bool(re.search(r"experiment|ablation|patching|attribution|logit", text)),
        "non_empty": len(text.strip()) > 0,
    }
    score = sum(checks.values()) / len(checks)
    return GraderResult(
        criterion="plan_well_formed", agent=AGENT, grader_type=GraderType.CODE,
        score=score, metrics=checks,
        detail=f"{sum(checks.values())}/{len(checks)} structural elements present",
    )


def target_model_appropriate(plan: str, expected_model_substrings: list[str]) -> GraderResult:
    """CODE variant: does the plan name a model matching the ground-truth target?
    (The open-ended 'is this an appropriate choice when several are valid' case is
    the LLM variant, graded elsewhere.)"""
    text = (plan or "").lower()
    hit = next((s for s in expected_model_substrings if s.lower() in text), None)
    return GraderResult(
        criterion="target_model_appropriate", agent=AGENT, grader_type=GraderType.CODE,
        score=1.0 if hit else 0.0,
        metrics={"matched": hit, "expected_any_of": expected_model_substrings},
        detail=f"named {hit!r}" if hit else f"named none of {expected_model_substrings}",
    )


_ARXIV_RE = re.compile(r"arxiv[:/]?\s*(\d{4}\.\d{4,5})", re.I)
_ARXIV_BARE_RE = re.compile(r"\b(\d{4}\.\d{4,5})\b")
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.I)


def citations_exist(plan: str, *, check_network: bool = True, timeout: float = 8.0) -> GraderResult:
    """Do the plan's citations resolve to real papers? Extracts arXiv IDs and DOIs
    and (optionally) resolves each. Catches fabricated citations; says nothing
    about whether the paper actually supports the claim (that's the LLM grader)."""
    text = plan or ""
    arxiv_ids = set(_ARXIV_RE.findall(text)) | set(_ARXIV_BARE_RE.findall(text))
    dois = set(_DOI_RE.findall(text))
    ids = [("arxiv", a) for a in arxiv_ids] + [("doi", d) for d in dois]

    if not ids:
        return GraderResult(
            criterion="citations_exist", agent=AGENT, grader_type=GraderType.CODE,
            score=None, metrics={"citations_found": 0}, needs_human=True,
            detail="no machine-checkable citation IDs (arXiv/DOI) found — needs human check",
        )
    if not check_network:
        return GraderResult(
            criterion="citations_exist", agent=AGENT, grader_type=GraderType.CODE,
            score=None, metrics={"citations_found": len(ids), "resolved": None},
            detail=f"{len(ids)} citation IDs extracted (network check disabled)",
        )

    resolved, unresolved = [], []
    for kind, cid in ids:
        url = (f"https://export.arxiv.org/abs/{cid}" if kind == "arxiv"
               else f"https://doi.org/{cid}")
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "seesaw-eval"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                (resolved if resp.status < 400 else unresolved).append(cid)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
            unresolved.append(cid)

    score = len(resolved) / len(ids)
    return GraderResult(
        criterion="citations_exist", agent=AGENT, grader_type=GraderType.CODE,
        score=score, metrics={"resolved": resolved, "unresolved": unresolved},
        detail=f"{len(resolved)}/{len(ids)} citations resolved",
    )


def tools_called_properly(trajectory: list[dict]) -> GraderResult:
    """Are Scout's tool calls well-formed — known tool name, dict args? Operates on
    the run trace, not the plan. Catches malformed search queries / bad wiring."""
    if not trajectory:
        return GraderResult(
            criterion="tools_called_properly", agent=AGENT, grader_type=GraderType.CODE,
            score=None, metrics={"n_calls": 0}, needs_human=True,
            detail="no tool calls in trajectory",
        )
    bad = []
    for call in trajectory:
        name, args = call.get("name"), call.get("args")
        if name not in SCOUT_TOOLS or not isinstance(args, dict):
            bad.append({"name": name, "args_type": type(args).__name__})
    score = 1 - len(bad) / len(trajectory)
    return GraderResult(
        criterion="tools_called_properly", agent=AGENT, grader_type=GraderType.CODE,
        score=score, metrics={"n_calls": len(trajectory), "malformed": bad},
        detail=f"{len(trajectory) - len(bad)}/{len(trajectory)} tool calls well-formed",
    )


def efficiency(usage: dict, *, thresholds: dict | None = None) -> GraderResult:
    """Operational cost of the run (tool calls, tokens, wall-clock, redundant
    searches). Not a quality signal — a regression guard. `usage` is collected by
    the caller; this just packages it and flags threshold breaches if given."""
    thresholds = thresholds or {}
    breaches = {k: usage.get(k) for k, lim in thresholds.items()
                if usage.get(k) is not None and usage[k] > lim}
    score = 0.0 if breaches else (1.0 if thresholds else None)
    return GraderResult(
        criterion="efficiency", agent=AGENT, grader_type=GraderType.CODE,
        score=score, metrics={"usage": usage, "breaches": breaches},
        detail=(f"within all {len(thresholds)} thresholds" if thresholds and not breaches
                else f"exceeded: {list(breaches)}" if breaches else "metrics recorded (no thresholds)"),
    )


# ── LLM graders ──────────────────────────────────────────────────────────────
def hypothesis_specificity(plan: str, reference_text: str = "") -> GraderResult:
    """Are the plan's hypotheses specific and falsifiable (name a concrete
    head/layer/direction + predicted effect), not vaguely on-topic? Core quality
    axis. Anchored 1-5 rubric to reduce judge variance."""
    from pydantic import BaseModel

    class Grade(BaseModel):
        score: int          # 1-5
        unable_to_grade: bool = False
        reasoning: str

    g: "Grade" = judge(
        "Grade the SPECIFICITY and falsifiability of the hypotheses in this mechanistic "
        "interpretability research plan. A hypothesis is specific if it names a concrete, "
        "falsifiable prediction (e.g. 'head L9H6 shows positive attribution toward the IO "
        "token') rather than a vague direction (e.g. 'middle-layer heads are probably "
        "involved'). Score 1-5: 5 = every hypothesis is concrete and testable; 1 = vague or "
        "unfalsifiable. If the plan is too garbled/off-topic to grade, set unable_to_grade.\n\n"
        f"REFERENCE (known-good target):\n{reference_text[:3000]}\n\n"
        f"PLAN:\n{(plan or '')[:3000]}",
        Grade,
    )
    if g.unable_to_grade:
        return GraderResult("hypothesis_specificity", AGENT, GraderType.LLM,
                            score=None, detail=f"unable to grade: {g.reasoning}", needs_human=True)
    return GraderResult("hypothesis_specificity", AGENT, GraderType.LLM,
                        score=g.score / 5, metrics={"raw_score": g.score}, detail=g.reasoning)


def groundedness(plan: str, retrieved_context: str) -> GraderResult:
    """Are the plan's factual claims traceable to what search actually returned, or
    hallucinated? Requires the retrieved context, not just the final plan."""
    from pydantic import BaseModel

    class Grade(BaseModel):
        grounded_fraction: float   # 0-1: share of checkable claims supported by context
        unsupported_claims: list[str]
        reasoning: str

    g: "Grade" = judge(
        "Below is a research plan and the search results the author actually retrieved. "
        "Judge how well the plan's factual claims are grounded in that retrieved context. "
        "Report grounded_fraction (0-1) and list any claims not supported by the context.\n\n"
        f"RETRIEVED CONTEXT:\n{(retrieved_context or '')[:4000]}\n\n"
        f"PLAN:\n{(plan or '')[:3000]}",
        Grade,
    )
    return GraderResult("groundedness", AGENT, GraderType.LLM,
                        score=max(0.0, min(1.0, g.grounded_fraction)),
                        metrics={"unsupported_claims": g.unsupported_claims}, detail=g.reasoning)


def citations_support_hypothesis(plan: str) -> GraderResult:
    """Do the cited papers actually support the hypotheses (not just exist / not just
    be on-topic)? The 'real paper, wrong claim' failure. LLM judge; humans audit."""
    from pydantic import BaseModel

    class Grade(BaseModel):
        supported_fraction: float
        mismatches: list[str]      # citation → claim it's used for but doesn't support
        reasoning: str

    g: "Grade" = judge(
        "In this research plan, check each citation against the specific claim it is cited "
        "for. A citation is 'supporting' only if the referenced work actually backs that "
        "claim — not merely shares the topic. Report supported_fraction (0-1) and list "
        "mismatches (a real or plausible paper cited for a claim it doesn't support).\n\n"
        f"PLAN:\n{(plan or '')[:4000]}",
        Grade,
    )
    return GraderResult("citations_support_hypothesis", AGENT, GraderType.LLM,
                        score=max(0.0, min(1.0, g.supported_fraction)),
                        metrics={"mismatches": g.mismatches}, detail=g.reasoning, needs_human=True)


def search_triggering(question: str, searched: bool) -> GraderResult:
    """Did Scout search when it should have (and not when it shouldn't)? `searched`
    is a fact from the trace; the judgment is whether the question warranted it."""
    from pydantic import BaseModel

    class Grade(BaseModel):
        should_search: bool
        reasoning: str

    g: "Grade" = judge(
        "For this mechanistic-interpretability research question, should the planner have "
        "run a literature/web search before answering, or could it reasonably answer from "
        "parametric knowledge? Decide should_search (true/false).\n\n"
        f"QUESTION: {question}",
        Grade,
    )
    correct = g.should_search == searched
    mode = ("over_triggered" if searched and not g.should_search
            else "under_triggered" if not searched and g.should_search else "ok")
    return GraderResult("search_triggering", AGENT, GraderType.LLM,
                        score=1.0 if correct else 0.0,
                        metrics={"searched": searched, "should_search": g.should_search, "mode": mode},
                        detail=f"{mode}: {g.reasoning}")


def tool_capability_honesty(plan: str, missing_capability: str) -> GraderResult:
    """For a question needing an unbuilt technique, does the plan honestly flag the
    gap, or silently force-fit an available tool? Behavioral probe."""
    from pydantic import BaseModel

    class Grade(BaseModel):
        acknowledges_gap: bool
        reasoning: str

    g: "Grade" = judge(
        f"This plan targets a question that genuinely needs '{missing_capability}', which the "
        "available toolset (logit_lens, attention_pattern, ablation, activation_patching, "
        "direct_logit_attribution) cannot properly provide. Does the plan honestly acknowledge "
        "this limitation (flags the ideal technique is unavailable / proposes only a partial "
        "approach / notes low confidence), or propose the available tools as if adequate?\n\n"
        f"PLAN:\n{(plan or '')[:3000]}",
        Grade,
    )
    return GraderResult("tool_capability_honesty", AGENT, GraderType.LLM,
                        score=1.0 if g.acknowledges_gap else 0.0,
                        metrics={"acknowledges_gap": g.acknowledges_gap}, detail=g.reasoning)


# ── Human grader ─────────────────────────────────────────────────────────────
def researcher_would_run(plan: str, question: str = "") -> GraderResult:
    """North star: would a real mech-interp researcher run this plan? Packages the
    artifact + rubric for an expert; also the yardstick to validate the LLM judges."""
    return GraderResult(
        "researcher_would_run", AGENT, GraderType.HUMAN, score=None, needs_human=True,
        detail="expert review required: rate plan usefulness 1-5 ('would you run this?')",
        metrics={"review_payload": {
            "question": question, "plan": plan,
            "rubric": "1 = would not run; 5 = would run as-is. Judge scientific merit, "
                      "not surface polish. Note the single biggest reason for the score.",
        }},
    )
