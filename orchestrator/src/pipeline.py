"""Seesaw pipeline: Scout → Lens → Quill with HITL checkpoints."""

from pathlib import Path

from .clients.lens_client import run_lens
from .clients.quill_client import run_quill
from .clients.scout_client import run_scout
from .hitl import checkpoint, show


def run_pipeline(
    research_question: str,
    skip_hitl: bool = False,
) -> dict:
    """Run the full Seesaw pipeline end-to-end.

    Stages:
        1. Scout   — research question → research plan
        2. HITL    — user reviews and approves the plan
        3. Lens    — research plan → experiment results
        4. HITL    — user reviews results before critique
        5. Quill   — results bundle → critique report + follow-ups

    Args:
        research_question: The mechanistic interpretability question to investigate.
        skip_hitl: If True, skip all HITL checkpoints (for automated runs).

    Returns:
        Dict with keys: plan_path, bundle_path, report_path, report.
    """
    print(f"\n🎯 Seesaw pipeline starting")
    print(f"   Question: {research_question!r}\n")

    # ── Stage 1: Scout ────────────────────────────────────────────────────────
    plan_path = run_scout(research_question)

    # ── HITL 1: Review the research plan ─────────────────────────────────────
    if not skip_hitl:
        plan_text = plan_path.read_text()
        proceed   = checkpoint(
            title    = "Scout Research Plan",
            content  = plan_text[:3_000] + ("\n...[truncated]" if len(plan_text) > 3_000 else ""),
            question = "Approve this plan and run Lens experiments?",
        )
        if not proceed:
            return {"plan_path": plan_path, "bundle_path": None, "report_path": None, "report": None}
    else:
        plan_text = plan_path.read_text()

    # ── Stage 2: Lens ─────────────────────────────────────────────────────────
    bundle, bundle_path = run_lens(plan_text)

    # ── HITL 2: Review experiment results ────────────────────────────────────
    if not skip_hitl:
        summary = _format_bundle_summary(bundle)
        proceed = checkpoint(
            title    = "Lens Experiment Results",
            content  = summary,
            question = "Send these results to Quill for critique?",
        )
        if not proceed:
            return {"plan_path": plan_path, "bundle_path": bundle_path, "report_path": None, "report": None}

    # ── Stage 3: Quill ────────────────────────────────────────────────────────
    report, report_path = run_quill(bundle_path)

    if not skip_hitl:
        show("Quill Critique Report", _format_report_summary(report))

    print(f"\n✅ Pipeline complete")
    print(f"   Plan    : {plan_path}")
    print(f"   Results : {bundle_path}")
    print(f"   Critique: {report_path}")

    return {
        "plan_path":   plan_path,
        "bundle_path": bundle_path,
        "report_path": report_path,
        "report":      report,
    }


def _format_bundle_summary(bundle: dict) -> str:
    lines = [
        f"Research question: {bundle['research_question']}",
        f"Model: {bundle['model_name']}",
        f"Results: {bundle['n_success']} succeeded, {bundle['n_failed']} failed",
        "",
    ]
    for r in bundle.get("results", []):
        status = "✅" if r["status"] == "success" else "❌"
        lines.append(f"{status} {r['name']} ({r['tool']})")
        if r.get("summary"):
            lines.append(f"   {r['summary'][:200]}")
    return "\n".join(lines)


def _format_report_summary(report) -> str:
    lines = [
        f"Overall assessment: {report.overall_assessment.upper()}",
        f"Summary: {report.overall_summary}",
        f"Coverage: {report.coverage_verdict}",
        "",
        f"Gaps identified: {len(report.gaps)}",
        f"Follow-ups suggested: {len(report.followups)}",
    ]
    if report.followups:
        lines.append("\nTop follow-ups:")
        for f in report.followups[:3]:
            lines.append(f"  [{f.priority}] {f.name} — {f.tool}")
    return "\n".join(lines)
