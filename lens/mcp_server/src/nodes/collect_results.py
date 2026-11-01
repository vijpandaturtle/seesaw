import json
from pathlib import Path

from ..config import OUTPUTS_DIR


def collect_results(state: dict) -> dict:
    """Bundle all experiment results and save to disk for Quill.

    Node 6 (final) in the Lens workflow.
    """
    print("\n📦 [collect_results] Bundling...")
    bundle = {
        "research_question": state["research_question"],
        "model_name":        state["model_name"],
        "results":           state["results"],
        "n_total":           len(state["results"]),
        "n_success":         sum(1 for r in state["results"] if r["status"] == "success"),
        "n_failed":          sum(1 for r in state["results"] if r["status"] != "success"),
    }
    output_dir = Path(state.get("output_dir") or OUTPUTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "results_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, default=str))
    print(f"   ✅ {bundle['n_success']}/{bundle['n_total']} succeeded → {bundle_path}")
    return {"bundle": bundle}
