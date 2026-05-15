from langchain_core.tools import tool

from ..config import OUTPUTS_DIR


@tool
def save_research_plan(plan: str, filename: str = "research_plan.md") -> str:
    """Save the final structured Research Plan to disk.
    Call this ONLY when you have gathered sufficient information and are ready
    to produce the final output. This signals the end of the Scout workflow.

    Args:
        plan: The complete research plan in markdown format
        filename: Output filename (default: research_plan.md)

    Returns:
        Confirmation message with the saved file path
    """
    output_path = OUTPUTS_DIR / filename
    output_path.write_text(plan, encoding="utf-8")
    return (
        f"✅ Research plan saved to: {output_path.resolve()}\n\n"
        f"Plan preview (first 500 chars):\n{plan[:500]}..."
    )
