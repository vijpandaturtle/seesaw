"""MCP tool registration for Quill.

Quill has no external tools of its own (see ../tools/) — it's a pure
critique agent that reasons over an ExperimentBundle. The one MCP tool
exposed here wraps the entire LangGraph workflow (critique → gaps →
follow-ups → save) as a single callable, since the workflow's stages are
not independently useful outside their sequence.
"""

import json

from fastmcp import FastMCP

from ..config import OUTPUTS_DIR
from ..models.schemas import ExperimentBundle, ExperimentResult
from ..workflows.critique_workflow import build_quill_graph


def register_mcp_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def critique_experiment_bundle(bundle_json: str, thread_id: str = "quill-mcp") -> str:
        """Run the full Quill critique workflow on a Lens results bundle.

        Assesses each experiment for validity and confounds, identifies
        gaps relative to the research question, generates concrete
        follow-up experiment specs Lens can execute directly, and saves
        a timestamped critique report (JSON + Markdown) to quill/outputs/.

        Args:
            bundle_json: JSON string of an ExperimentBundle, as produced
                by Lens (keys: research_question, model_name, results).
            thread_id: LangGraph checkpoint thread ID.

        Returns:
            JSON string of the CritiqueReport.
        """
        raw     = json.loads(bundle_json)
        results = [ExperimentResult(**r) for r in raw["results"]]
        bundle  = ExperimentBundle(
            research_question=raw["research_question"],
            model_name=raw["model_name"],
            results=results,
        )

        graph  = build_quill_graph()
        config = {"configurable": {"thread_id": thread_id}}
        final  = graph.invoke({"bundle": bundle, "output_dir": OUTPUTS_DIR}, config=config)

        report = final.get("critique_report")
        if report is None:
            return json.dumps({"error": "Quill workflow completed without producing a report"})

        return json.dumps(report.to_dict())
