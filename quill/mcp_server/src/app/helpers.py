from .schemas import CritiqueReport, ExperimentBundle, ExperimentResult


def format_result_for_context(result: ExperimentResult) -> str:
    """Format a single ExperimentResult as a markdown section for LLM context."""
    lines = [
        f"### Experiment: {result.name}",
        f"- **Tool**: `{result.tool}`",
        f"- **Model**: {result.model_name}",
        f"- **Status**: {result.status}",
        f"- **Prompts used**: {len(result.prompts)} (e.g. `{result.prompts[0][:80] if result.prompts else 'N/A'}`)",
        "",
        f"**Lens summary**: {result.summary or '(no summary)'}",
    ]
    if result.data:
        lines += ["", "**Key measurements**:"]
        for k, v in list(result.data.items())[:8]:
            lines.append(f"  - `{k}`: {str(v)[:200]}")
    if result.error:
        lines.append(f"**Error**: {result.error}")
    return "\n".join(lines)


def bundle_to_context(bundle: ExperimentBundle) -> str:
    """Render the full ExperimentBundle as markdown for LLM prompts."""
    parts = [
        f"Research question: {bundle.research_question}",
        f"Model: {bundle.model_name}",
        f"Experiments: {len(bundle.successful)} successful, {len(bundle.failed)} failed",
        "",
    ]
    parts += [format_result_for_context(r) for r in bundle.successful]

    if bundle.failed:
        parts.append("\n### Failed experiments")
        for r in bundle.failed:
            parts.append(
                f"- `{r.name}` ({r.tool}): {r.status} — {r.error or 'no error message'}"
            )

    return "\n\n".join(parts)


def render_critique_markdown(report: CritiqueReport) -> str:
    """Render a CritiqueReport to human-readable markdown."""
    SEVERITY_EMOJI = {"critical": "🔴", "important": "🟡", "minor": "🟢"}
    VALIDITY_EMOJI = {"strong": "✅", "moderate": "⚠️", "weak": "❌"}
    PRIORITY_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}

    lines = [
        "# Quill Critique Report",
        "",
        f"> **Research question**: {report.research_question}",
        f"> **Model**: `{report.model_name}`",
        f"> **Generated**: {report.generated_at[:19]}",
        "",
        "---",
        "",
        f"## Overall Assessment: {report.overall_assessment.upper()}",
        "",
        report.overall_summary,
        "",
        f"**Coverage**: {report.coverage_verdict}",
        "",
        "---",
        "",
        "## Experiment Critiques",
    ]

    for c in report.critiques:
        supported = "conclusions supported" if c.conclusions_supported else "**conclusions NOT supported**"
        lines += [
            "",
            f"### {VALIDITY_EMOJI[c.validity]} {c.experiment_name}",
            f"- **Validity**: {c.validity} | {supported}",
        ]
        if c.positive_aspects:
            lines.append("- **Strengths**: " + "; ".join(c.positive_aspects))
        if c.issues:
            lines.append("- **Issues**:")
            for issue in c.issues:
                lines.append(f"  - {issue}")
        if c.alternative_explanations:
            lines.append("- **Alternative explanations**:")
            for alt in c.alternative_explanations:
                lines.append(f"  - {alt}")

    lines += [
        "",
        "---",
        "",
        "## Research Gaps",
    ]

    if not report.gaps:
        lines.append("\nNo significant gaps identified.")
    else:
        for g in sorted(
            report.gaps,
            key=lambda x: ["critical", "important", "minor"].index(x.severity),
        ):
            lines += [
                "",
                f"### {SEVERITY_EMOJI[g.severity]} {g.description}",
                f"- **Severity**: {g.severity}",
                f"- **Why needed**: {g.why_needed}",
                f"- **Suggested tool**: `{g.suggested_tool}`",
            ]

    lines += [
        "",
        "---",
        "",
        "## Suggested Follow-Up Experiments",
        "",
        "*These specs can be passed directly to Lens for execution.*",
    ]

    if not report.followups:
        lines.append("\nNo follow-up experiments suggested.")
    else:
        for fu in sorted(
            report.followups,
            key=lambda x: ["high", "medium", "low"].index(x.priority),
        ):
            lines += [
                "",
                f"### {PRIORITY_EMOJI[fu.priority]} {fu.name}",
                f"- **Tool**: `{fu.tool}`",
                f"- **Priority**: {fu.priority}",
                f"- **Rationale**: {fu.rationale}",
                f"- **What to measure**: {fu.what_to_measure}",
                f"- **Hypothesis**: {fu.hypothesis}",
            ]

    return "\n".join(lines)
