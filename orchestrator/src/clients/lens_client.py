"""Client for the Lens experiment-execution workflow."""

from pathlib import Path

from lens.mcp_server.src.config.settings import OUTPUTS_DIR
from lens.mcp_server.src.workflows.lens_workflow import build_lens_graph


def run_lens(
    research_plan: str,
    thread_id: str = "lens-1",
) -> tuple[dict, Path]:
    """Run the Lens workflow on a research plan.

    Parses the plan, loads the model, runs each experiment, interprets
    results, and saves a results bundle JSON to lens/outputs/.

    Args:
        research_plan: Full text of the Scout research plan.
        thread_id: LangGraph thread ID for checkpointing.

    Returns:
        Tuple of (bundle dict, path to results_bundle.json).

    Raises:
        RuntimeError: If the workflow completes without a bundle.
    """
    graph  = build_lens_graph()
    config = {"configurable": {"thread_id": thread_id}}

    print("🔬 Lens starting experiments...")

    final = graph.invoke(
        {"research_plan": research_plan},
        config=config,
    )

    bundle = final.get("bundle")
    if bundle is None:
        raise RuntimeError("Lens workflow completed but produced no bundle.")

    bundle_path = OUTPUTS_DIR / "results_bundle.json"
    print(
        f"✅ Lens done — {bundle['n_success']}/{bundle['n_total']} experiments succeeded"
        f" → {bundle_path}"
    )
    return bundle, bundle_path
