from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..app.helpers import bundle_to_context
from ..models.schemas import GapsOutput
from ..config import (
    ANTHROPIC_API_KEY,
    IDENTIFY_GAPS_SYSTEM,
    QUILL_MODEL,
    QUILL_TEMPERATURE,
)


def identify_gaps(state: dict) -> dict:
    """Identify missing experiments relative to what the research question demands.

    Node 2 in the Quill workflow.
    """
    bundle           = state["bundle"]
    critiques_output = state["critiques_output"]
    print("\n🕳️  [identify_gaps] Identifying missing experiments...")

    model = ChatAnthropic(
        model=QUILL_MODEL,
        temperature=QUILL_TEMPERATURE,
        api_key=ANTHROPIC_API_KEY,
    )
    structured = model.with_structured_output(GapsOutput)

    ctx         = bundle_to_context(bundle)
    tools_used  = list({r.tool for r in bundle.successful})
    critique_summary = "\n".join(
        f"- {c.experiment_name}: {c.validity}, issues: {'; '.join(c.issues[:2])}"
        for c in critiques_output.critiques
    )

    prompt = (
        f"Research question: {bundle.research_question}\n\n"
        f"Tools used so far: {tools_used}\n\n"
        f"Experiment critiques (summary):\n{critique_summary}\n\n"
        f"Full experiment data:\n{ctx}\n\n"
        f"What experiments are missing? What would a thorough investigation of this research "
        f"question require that wasn't done?"
    )

    output: GapsOutput = structured.invoke([
        SystemMessage(content=IDENTIFY_GAPS_SYSTEM),
        HumanMessage(content=prompt),
    ])

    critical  = [g for g in output.gaps if g.severity == "critical"]
    important = [g for g in output.gaps if g.severity == "important"]
    minor     = [g for g in output.gaps if g.severity == "minor"]
    print(
        f"  → {len(output.gaps)} gaps: "
        f"{len(critical)} critical, {len(important)} important, {len(minor)} minor"
    )
    for g in output.gaps:
        print(f"     [{g.severity}] {g.description[:80]}")

    return {"gaps_output": output}
