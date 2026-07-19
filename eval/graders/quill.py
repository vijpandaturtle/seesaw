"""Quill graders — grade a CritiqueReport against a bundle Quill reviewed.

Quill has no ground-truth paper; you evaluate JUDGMENT. The method is to feed
bundles with known/planted properties and check Quill reacts correctly, so the
human 'label author' role dominates here — the CODE graders below run on
human-authored fixtures (strong/weak labels, planted-flaw lists).

CritiqueReport shape (quill/mcp_server/src/models/schemas.py):
  overall_assessment : "strong" | "moderate" | "weak"   (three-way, not binary)
  gaps               : [ResearchGap(description, severity, why_needed, suggested_tool)]
  followups          : [FollowUpSpec(name, tool, rationale, what_to_measure, hypothesis, priority)]
  critiques          : [ExperimentCritique(...)]

Graders read fields via base.field(), accepting the dataclass or a dict.

`statistical_validity` lives here (not under Lens): rigor-critique is Quill's job.
Lens grades only whether its numbers are right and honestly described.
"""

from __future__ import annotations

from .base import GraderResult, GraderType, field, judge

AGENT = "quill"

# Three-way assessment scale, for ordinal miscalibration distance.
_ASSESSMENT_RANK = {"weak": 0, "moderate": 1, "strong": 2}

# Mirrors lens TOOL_REGISTRY — a follow-up is only executable if its tool exists.
VALID_LENS_TOOLS = {
    "logit_lens", "attention_pattern", "direct_logit_attribution",
    "ablation", "activation_patching",
}


# ── Code graders ─────────────────────────────────────────────────────────────
def calibration(report, expected_assessment: str) -> GraderResult:
    """Does Quill's overall_assessment match the human-authored gold label? Exact
    match scores 1.0; adjacent (moderate↔strong) scores 0.5 via ordinal distance, so
    a near-miss reads differently from calling a weak run strong."""
    observed = field(report, "overall_assessment")
    exact = observed == expected_assessment
    o, e = _ASSESSMENT_RANK.get(observed), _ASSESSMENT_RANK.get(expected_assessment)
    dist = abs(o - e) if o is not None and e is not None else 2
    score = 1.0 if exact else (0.5 if dist == 1 else 0.0)
    return GraderResult(
        "calibration", AGENT, GraderType.CODE, score=score,
        metrics={"observed": observed, "expected": expected_assessment, "ordinal_distance": dist},
        detail=f"observed={observed!r} expected={expected_assessment!r} (dist={dist})",
    )


def false_positive_rate(report) -> GraderResult:
    """On a genuinely-good bundle, Quill should raise ~no gaps. CODE half: count the
    gaps (target 0). Pair with false_positives_spurious() for the LLM half that
    judges whether any raised gaps are actually spurious."""
    gaps = field(report, "gaps", []) or []
    critical = [g for g in gaps if field(g, "severity") in ("critical", "important")]
    # Score decays with the number of serious gaps invented on a clean run.
    score = 1.0 / (1 + len(critical))
    return GraderResult(
        "false_positive_rate", AGENT, GraderType.CODE, score=score,
        metrics={"n_gaps": len(gaps), "n_critical_or_important": len(critical)},
        detail=f"{len(gaps)} gaps raised on a good bundle ({len(critical)} serious)",
    )


def followup_executable(report) -> GraderResult:
    """Are Quill's follow-up specs runnable by Lens — valid tool name + the fields an
    ExperimentSpec needs? Strong signal; literally closes the Quill→Lens loop."""
    followups = field(report, "followups", []) or []
    if not followups:
        return GraderResult("followup_executable", AGENT, GraderType.CODE, score=None,
                            metrics={"n_followups": 0}, detail="no follow-ups proposed")
    bad = []
    for f in followups:
        tool = field(f, "tool")
        ok = tool in VALID_LENS_TOOLS and bool(field(f, "what_to_measure")) and bool(field(f, "name"))
        if not ok:
            bad.append({"name": field(f, "name"), "tool": tool,
                        "tool_valid": tool in VALID_LENS_TOOLS})
    score = 1 - len(bad) / len(followups)
    return GraderResult(
        "followup_executable", AGENT, GraderType.CODE, score=score,
        metrics={"n_followups": len(followups), "invalid": bad},
        detail=f"{len(followups) - len(bad)}/{len(followups)} follow-ups are executable",
    )


def stability(assessments: list[str]) -> GraderResult:
    """Run the same bundle N times → same verdict? Quill is non-deterministic; high
    variance here undermines every other Quill metric. Score = modal agreement rate."""
    n = len(assessments)
    if n < 2:
        return GraderResult("stability", AGENT, GraderType.CODE, score=None, needs_human=True,
                            detail="need >=2 repeat runs to measure stability")
    modal = max(set(assessments), key=assessments.count)
    agreement = assessments.count(modal) / n
    return GraderResult(
        "stability", AGENT, GraderType.CODE, score=agreement,
        metrics={"n_runs": n, "modal_verdict": modal, "distribution":
                 {a: assessments.count(a) for a in set(assessments)}},
        detail=f"{assessments.count(modal)}/{n} runs agreed on {modal!r}",
    )


# ── LLM graders ──────────────────────────────────────────────────────────────
def gap_precision(report, bundle) -> GraderResult:
    """Are the gaps Quill flagged real problems in THIS bundle, not hallucinated?
    precision = real / flagged. Subtle cases get a human audit."""
    from pydantic import BaseModel

    class Grade(BaseModel):
        real_count: int
        flagged_count: int
        hallucinated: list[str]
        reasoning: str

    gaps = [field(g, "description") for g in (field(report, "gaps", []) or [])]
    if not gaps:
        return GraderResult("gap_precision", AGENT, GraderType.LLM, score=None,
                            detail="no gaps flagged")
    g: "Grade" = judge(
        "Here is a mechanistic-interpretability experiment bundle and the gaps a reviewer "
        "flagged. For each flagged gap, decide whether it is a REAL problem actually present "
        "in this bundle (not a hallucinated or generic critique). Report real_count, "
        "flagged_count, and list any hallucinated gaps.\n\n"
        f"BUNDLE:\n{_bundle_digest(bundle)}\n\nFLAGGED GAPS:\n{gaps}",
        Grade,
    )
    denom = g.flagged_count or len(gaps)
    return GraderResult("gap_precision", AGENT, GraderType.LLM, score=g.real_count / denom,
                        metrics={"real": g.real_count, "flagged": denom, "hallucinated": g.hallucinated},
                        detail=g.reasoning, needs_human=True)


def gap_recall(report, planted_flaws: list[str]) -> GraderResult:
    """Did Quill catch the deliberately planted flaws? recall = caught / planted.
    Highest-leverage, most human-intensive fixture to build (seeded known flaws)."""
    from pydantic import BaseModel

    class Grade(BaseModel):
        caught: list[str]      # which planted flaws a gap covers
        missed: list[str]
        reasoning: str

    if not planted_flaws:
        return GraderResult("gap_recall", AGENT, GraderType.LLM, score=None,
                            detail="no planted flaws supplied")
    gaps = [field(g, "description") for g in (field(report, "gaps", []) or [])]
    g: "Grade" = judge(
        "This bundle was constructed with a known set of flaws (below). Given the reviewer's "
        "flagged gaps, decide which planted flaws are covered by at least one gap (caught) and "
        "which are not (missed).\n\n"
        f"PLANTED FLAWS:\n{planted_flaws}\n\nREVIEWER'S GAPS:\n{gaps}",
        Grade,
    )
    return GraderResult("gap_recall", AGENT, GraderType.LLM, score=len(g.caught) / len(planted_flaws),
                        metrics={"caught": g.caught, "missed": g.missed, "n_planted": len(planted_flaws)},
                        detail=g.reasoning)


def false_positives_spurious(report) -> GraderResult:
    """LLM half of the good-bundle check: of the gaps raised, how many are spurious
    (not a genuine methodological problem)? Pairs with false_positive_rate()."""
    from pydantic import BaseModel

    class Grade(BaseModel):
        spurious: list[str]
        total: int
        reasoning: str

    gaps = [field(g, "description") for g in (field(report, "gaps", []) or [])]
    if not gaps:
        return GraderResult("false_positives_spurious", AGENT, GraderType.LLM, score=1.0,
                            metrics={"n_gaps": 0}, detail="no gaps raised — no false positives")
    g: "Grade" = judge(
        "This bundle is a genuinely strong run. Of the reviewer's flagged gaps, list any that "
        "are spurious (not a real methodological problem — a manufactured or trivial critique).\n\n"
        f"FLAGGED GAPS:\n{gaps}",
        Grade,
    )
    denom = g.total or len(gaps)
    return GraderResult("false_positives_spurious", AGENT, GraderType.LLM,
                        score=1 - len(g.spurious) / denom,
                        metrics={"spurious": g.spurious, "total": denom}, detail=g.reasoning)


def followup_relevant(report) -> GraderResult:
    """Do the follow-ups actually address the gaps Quill identified?"""
    from pydantic import BaseModel

    class Grade(BaseModel):
        relevant_fraction: float
        irrelevant: list[str]
        reasoning: str

    gaps = [field(g, "description") for g in (field(report, "gaps", []) or [])]
    followups = [{"name": field(f, "name"), "rationale": field(f, "rationale"),
                  "measures": field(f, "what_to_measure")}
                 for f in (field(report, "followups", []) or [])]
    if not followups:
        return GraderResult("followup_relevant", AGENT, GraderType.LLM, score=None,
                            detail="no follow-ups proposed")
    g: "Grade" = judge(
        "Given the identified gaps and the reviewer's proposed follow-up experiments, what "
        "fraction of the follow-ups actually address a stated gap? List any that don't.\n\n"
        f"GAPS:\n{gaps}\n\nFOLLOW-UPS:\n{followups}",
        Grade,
    )
    return GraderResult("followup_relevant", AGENT, GraderType.LLM,
                        score=max(0.0, min(1.0, g.relevant_fraction)),
                        metrics={"irrelevant": g.irrelevant}, detail=g.reasoning)


def statistical_validity(bundle) -> GraderResult:
    """Rigor of the RUN itself — baselines/controls present, no over-generalization
    from a single prompt. Reassigned here from Lens so rigor-critique is owned
    solely by Quill. Pairs naturally with gap_recall on seeded-control bundles."""
    from pydantic import BaseModel

    class Grade(BaseModel):
        score: int             # 1-5 rigor
        issues: list[str]      # missing baseline/control, over-generalization, ...
        reasoning: str

    g: "Grade" = judge(
        "Assess the experimental rigor of this mechanistic-interpretability bundle. Did it use "
        "appropriate baselines and controls? Does it over-generalize from a single prompt or "
        "a single run? Score 1-5 (5 = rigorous) and list concrete rigor issues.\n\n"
        f"BUNDLE:\n{_bundle_digest(bundle)}",
        Grade,
    )
    return GraderResult("statistical_validity", AGENT, GraderType.LLM, score=g.score / 5,
                        metrics={"raw_score": g.score, "issues": g.issues}, detail=g.reasoning,
                        needs_human=True)


# ── Human grader ─────────────────────────────────────────────────────────────
def matches_expert(report, bundle) -> GraderResult:
    """North star: does Quill's critique match an expert peer reviewer's? Also the
    audit that validates the LLM gap-judges above. Packaged for blind human review."""
    return GraderResult(
        "matches_expert", AGENT, GraderType.HUMAN, score=None, needs_human=True,
        detail="expert review required: agreement / Cohen's kappa vs Quill on the same bundle",
        metrics={"review_payload": {
            "bundle": _bundle_digest(bundle),
            "quill_assessment": field(report, "overall_assessment"),
            "quill_gaps": [field(g, "description") for g in (field(report, "gaps", []) or [])],
            "rubric": "Independently assess this bundle (strong/moderate/weak + gaps), THEN "
                      "compare to Quill's. Record agreement on assessment and gap overlap.",
        }},
    )


# ── helpers ──────────────────────────────────────────────────────────────────
def _bundle_digest(bundle) -> str:
    """Compact text view of an ExperimentBundle for a judge prompt."""
    rq = field(bundle, "research_question", "")
    results = field(bundle, "results", []) or []
    lines = [f"Research question: {rq}", f"Model: {field(bundle, 'model_name', '')}",
             f"{len(results)} experiment(s):"]
    for r in results:
        lines.append(f"  - [{field(r, 'tool')}] {field(r, 'name')}: {field(r, 'summary', '')[:300]}")
    return "\n".join(lines)
