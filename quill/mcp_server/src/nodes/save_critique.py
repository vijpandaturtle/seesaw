import json
from datetime import datetime

from ..app.helpers import render_critique_markdown
from ..models.schemas import CritiqueReport
from ..config import OUTPUTS_DIR


def save_critique(state: dict) -> dict:
    """Assemble the CritiqueReport and save it as JSON + Markdown.

    Node 4 (final) in the Quill workflow.
    """
    bundle           = state["bundle"]
    critiques_output = state["critiques_output"]
    gaps_output      = state["gaps_output"]
    followups_output = state["followups_output"]
    output_dir       = state.get("output_dir", OUTPUTS_DIR)

    print("\n💾 [save_critique] Assembling and saving critique report...")

    report = CritiqueReport(
        research_question=bundle.research_question,
        model_name=bundle.model_name,
        overall_assessment=critiques_output.overall_assessment,
        overall_summary=critiques_output.overall_summary,
        coverage_verdict=gaps_output.coverage_verdict,
        critiques=critiques_output.critiques,
        gaps=gaps_output.gaps,
        followups=followups_output.followups,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"critique_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    md_path = output_dir / f"critique_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(render_critique_markdown(report))

    print(f"  ✅ JSON  → {json_path}")
    print(f"  ✅ MD    → {md_path}")

    return {
        "critique_report": report,
        "critique_path":   md_path,
    }
