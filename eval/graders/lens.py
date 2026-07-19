"""Lens graders — grade an ExperimentResult / ExperimentBundle.

Lens has two eval surfaces (see EVAL_MATRIX):
  Layer 1 — the numeric tools (deterministic forward passes): correctness,
            robustness, schema validity, plan→execution fidelity, cost. All CODE.
  Layer 2 — the LLM interpretation written on top of each number: prose↔data
            consistency (CODE) and faithfulness / causal-honesty (LLM).

Graders read fields via base.field(), so they accept a live ExperimentResult
dataclass OR a deserialized dict. ExperimentResult.data shape produced by the
tools, e.g. run_ablation / run_direct_logit_attribution:
    {"top_heads": [["L9H6", 4.1], ["L9H9", 3.8], ...], "baseline_ld": 3.36, ...}
"""

from __future__ import annotations

import re

from .base import GraderResult, GraderType, field, judge

AGENT = "lens"

# Mirrors lens/mcp_server/src/tools/__init__.py::TOOL_REGISTRY. Flat constant so
# this module needn't import the tools package (which pulls in torch).
VALID_LENS_TOOLS = {
    "logit_lens", "attention_pattern", "direct_logit_attribution",
    "ablation", "activation_patching",
}


def _top_head_names(data: dict, n: int = 5) -> list[str]:
    """Pull the top-n head labels ('L9H6') out of an ExperimentResult.data."""
    return [h for h, _ in (data.get("top_heads") or [])[:n]]


# ── Layer 1: numeric tools (CODE) ────────────────────────────────────────────
def numeric_correctness(result, expected_heads: list[str], *,
                        top_n: int = 5, baseline_range: tuple[float, float] | None = None) -> GraderResult:
    """Did the tool recover the known circuit — expected heads in the top-n, and
    (optionally) baseline effect size in range? The crown jewel; exact against
    ground truth. For InterpBench models the expected set is known by construction."""
    data = field(result, "data", {}) or {}
    top = _top_head_names(data, top_n)
    heads_hit = sorted(set(expected_heads) & set(top))
    metrics = {"top_n": top, "expected": expected_heads, "heads_found": heads_hit}
    score_parts = [1.0 if heads_hit else 0.0]

    if baseline_range is not None:
        ld = data.get("baseline_ld")
        in_range = ld is not None and baseline_range[0] <= ld <= baseline_range[1]
        metrics["baseline_ld"] = ld
        metrics["baseline_in_range"] = in_range
        score_parts.append(1.0 if in_range else 0.0)

    return GraderResult(
        "numeric_correctness", AGENT, GraderType.CODE, score=sum(score_parts) / len(score_parts),
        metrics=metrics,
        detail=f"found {heads_hit or 'none'} of expected {expected_heads} in top-{top_n}",
    )


def robustness(per_variation_pass: list[bool]) -> GraderResult:
    """pass@k / pass^k across equivalent prompts (name pairs, occupations). A finding
    must hold across variations, not one fixed sentence. Score = pass^k (the strict,
    trust-it-unattended number); pass@k reported alongside."""
    k = len(per_variation_pass)
    if k == 0:
        return GraderResult("robustness", AGENT, GraderType.CODE, score=None, needs_human=True,
                            detail="no variations supplied")
    n = sum(per_variation_pass)
    pass_at_k, pass_hat_k = n > 0, n == k
    return GraderResult(
        "robustness", AGENT, GraderType.CODE, score=1.0 if pass_hat_k else 0.0,
        metrics={"k": k, "n_pass": n, "pass_at_k": pass_at_k, "pass_hat_k": pass_hat_k},
        detail=f"{n}/{k} variations passed (pass@{k}={pass_at_k}, pass^{k}={pass_hat_k})",
    )


def tool_execution_valid(result) -> GraderResult:
    """Did the tool run cleanly and return a well-formed ExperimentResult?
    Reliability floor beneath every other Lens metric."""
    status = field(result, "status", "success")
    checks = {
        "status_success": status == "success",
        "has_tool": bool(field(result, "tool")),
        "has_data": bool(field(result, "data")),
        "no_error": field(result, "error") in (None, ""),
    }
    return GraderResult(
        "tool_execution_valid", AGENT, GraderType.CODE, score=sum(checks.values()) / len(checks),
        metrics={**checks, "status": status, "error": field(result, "error")},
        detail=f"status={status}; {sum(checks.values())}/{len(checks)} checks ok",
    )


def plan_execution_fidelity(planned: list[dict], executed: list) -> GraderResult:
    """Did Lens run the experiments the plan specified, with the right tool/prompts?
    Catches 'planned activation_patching, ran only DLA'. Matches on (tool, prompts)."""
    def key(spec):
        return (field(spec, "tool"), tuple(field(spec, "prompts", []) or []))

    executed_keys = {key(e) for e in executed}
    matched = [p for p in planned if key(p) in executed_keys]
    missing = [field(p, "name", field(p, "tool")) for p in planned if key(p) not in executed_keys]
    score = len(matched) / len(planned) if planned else None
    return GraderResult(
        "plan_execution_fidelity", AGENT, GraderType.CODE, score=score,
        metrics={"n_planned": len(planned), "n_executed": len(executed), "unrun": missing},
        detail=f"{len(matched)}/{len(planned)} planned experiments executed as specified",
    )


def operational_cost(metrics: dict, *, thresholds: dict | None = None) -> GraderResult:
    """Latency / peak memory / model-load time / interpretation tokens. Regression
    guard, not quality."""
    thresholds = thresholds or {}
    breaches = {k: metrics.get(k) for k, lim in thresholds.items()
                if metrics.get(k) is not None and metrics[k] > lim}
    score = 0.0 if breaches else (1.0 if thresholds else None)
    return GraderResult(
        "operational_cost", AGENT, GraderType.CODE, score=score,
        metrics={"cost": metrics, "breaches": breaches},
        detail=f"exceeded {list(breaches)}" if breaches else "within thresholds" if thresholds else "recorded",
    )


# ── Layer 2a: prose ↔ data consistency (CODE) ────────────────────────────────
_NUM_RE = re.compile(r"-?\d+\.\d+")


def _all_numeric_values(obj) -> list[float]:
    """Flatten every float appearing anywhere in an ExperimentResult.data."""
    out: list[float] = []
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out += _all_numeric_values(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _all_numeric_values(v)
    return out


def prose_data_consistency(result, *, tol: float = 0.05) -> GraderResult:
    """Do the decimal numbers quoted in the summary match values actually present in
    `data`? Cheap; catches the worst interpretation-layer failure — a summary that
    states figures the tool never produced. Unmatched numbers are flagged, not
    auto-failed (a number may be a legitimate derived quantity)."""
    summary = field(result, "summary", "") or ""
    data = field(result, "data", {}) or {}
    quoted = [float(x) for x in _NUM_RE.findall(summary)]
    if not quoted:
        return GraderResult("prose_data_consistency", AGENT, GraderType.CODE, score=None,
                            metrics={"quoted": []}, detail="no decimal figures quoted in summary")
    data_vals = _all_numeric_values(data)
    unmatched = [q for q in quoted if not any(abs(q - d) <= max(tol, tol * abs(d)) for d in data_vals)]
    score = 1 - len(unmatched) / len(quoted)
    return GraderResult(
        "prose_data_consistency", AGENT, GraderType.CODE, score=score,
        metrics={"quoted": quoted, "unmatched": unmatched},
        detail=f"{len(quoted) - len(unmatched)}/{len(quoted)} quoted figures match data",
    )


# ── Layer 2b: interpretation judgment (LLM) ──────────────────────────────────
def interpretation_faithfulness(result) -> GraderResult:
    """Does the summary describe the numbers accurately — neither over- nor
    under-claiming what the data supports? Main judge for the interpretation layer."""
    from pydantic import BaseModel

    class Grade(BaseModel):
        verdict: str            # "accurate" | "over_claims" | "under_claims"
        reasoning: str

    g: "Grade" = judge(
        "Below is a mechanistic-interpretability tool result: raw numeric `data` and the "
        "natural-language `summary` written about it. Judge whether the summary faithfully "
        "reflects the data — verdict one of: accurate, over_claims (asserts more than the "
        "numbers support), under_claims (misses a clear signal in the numbers).\n\n"
        f"DATA: {field(result, 'data', {})}\n\nSUMMARY: {field(result, 'summary', '')}",
        Grade,
    )
    return GraderResult(
        "interpretation_faithfulness", AGENT, GraderType.LLM,
        score=1.0 if g.verdict == "accurate" else 0.0,
        metrics={"verdict": g.verdict}, detail=g.reasoning,
    )


def causal_correlational_honesty(result) -> GraderResult:
    """Does the summary reserve causal language ('necessary', 'causes') for causal
    tools (ablation, activation_patching), rather than reading causation into DLA /
    attention (which are correlational)? Discipline-specific failure mode."""
    from pydantic import BaseModel

    class Grade(BaseModel):
        violates: bool          # True if it claims causation from a correlational tool
        reasoning: str

    g: "Grade" = judge(
        "A summary should only make causal claims ('X is necessary/causes Y') when backed by a "
        "causal tool (ablation, activation_patching). direct_logit_attribution and "
        "attention_pattern are correlational — they show association, not necessity. Given the "
        "tool used and the summary, does the summary make an unsupported causal claim?\n\n"
        f"TOOL: {field(result, 'tool')}\n\nSUMMARY: {field(result, 'summary', '')}",
        Grade,
    )
    return GraderResult(
        "causal_correlational_honesty", AGENT, GraderType.LLM,
        score=0.0 if g.violates else 1.0,
        metrics={"violates": g.violates}, detail=g.reasoning,
    )
