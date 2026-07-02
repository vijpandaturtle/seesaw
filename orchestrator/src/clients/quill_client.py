"""Client for the Quill critique workflow."""

from pathlib import Path

from quill.mcp_server.src.config.settings import OUTPUTS_DIR
from quill.mcp_server.src.models.schemas import CritiqueReport, ExperimentBundle
from quill.mcp_server.src.workflows.critique_workflow import build_quill_graph


def run_quill(
    bundle_path: Path,
    thread_id: str = "quill-1",
) -> tuple[CritiqueReport, Path]:
    """Run the Quill critique workflow on a Lens results bundle.

    Loads the bundle from disk, runs critique → gap analysis →
    follow-up generation → save, and writes the report to quill/outputs/.

    Args:
        bundle_path: Path to the results_bundle.json produced by Lens.
        thread_id: LangGraph thread ID for checkpointing.

    Returns:
        Tuple of (CritiqueReport, path to the saved critique markdown file).
        save_critique writes timestamped critique_<YYYYMMDD_HHMMSS>.json/.md
        files, so the returned path is whatever the workflow actually wrote —
        not a fixed filename.

    Raises:
        FileNotFoundError: If bundle_path does not exist.
        RuntimeError: If the workflow completes without a report.
    """
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found at {bundle_path}")

    bundle = ExperimentBundle.from_json(bundle_path)
    graph  = build_quill_graph()
    config = {"configurable": {"thread_id": thread_id}}

    print(f"✍️  Quill critiquing {len(bundle.results)} experiments...")

    final = graph.invoke(
        {"bundle": bundle, "output_dir": OUTPUTS_DIR},
        config=config,
    )

    report = final.get("critique_report")
    if report is None:
        raise RuntimeError("Quill workflow completed but produced no critique report.")

    report_path = final["critique_path"]
    print(f"✅ Quill done — assessment: {report.overall_assessment} → {report_path}")
    return report, report_path
