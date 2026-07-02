from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..models.schemas import FollowUpsOutput
from ..config import (
    ANTHROPIC_API_KEY,
    GENERATE_FOLLOWUPS_SYSTEM,
    QUILL_MODEL,
    QUILL_TEMPERATURE,
)


def generate_followups(state: dict) -> dict:
    """Design concrete follow-up experiment specs that Lens can execute.

    Node 3 in the Quill workflow.
    """
    bundle           = state["bundle"]
    critiques_output = state["critiques_output"]
    gaps_output      = state["gaps_output"]
    print("\n🧪 [generate_followups] Designing follow-up experiments...")

    model = ChatAnthropic(
        model=QUILL_MODEL,
        temperature=QUILL_TEMPERATURE,
        api_key=ANTHROPIC_API_KEY,
    )
    structured = model.with_structured_output(FollowUpsOutput)

    gaps_text = "\n".join(
        f"- [{g.severity}] {g.description} → needs: {g.suggested_tool} ({g.why_needed})"
        for g in gaps_output.gaps
    )
    weak_experiments = [
        c for c in critiques_output.critiques
        if c.validity in ("weak", "moderate") or not c.conclusions_supported
    ]
    weak_text = "\n".join(
        f"- {c.experiment_name}: {'; '.join(c.issues[:2])}"
        for c in weak_experiments
    ) or "(none — all experiments are sound)"

    prompt = (
        f"Research question: {bundle.research_question}\n"
        f"Model: {bundle.model_name}\n\n"
        f"Identified gaps:\n{gaps_text}\n\n"
        f"Experiments needing follow-up (weak or unsupported conclusions):\n{weak_text}\n\n"
        f"Design concrete follow-up experiments that Lens can run to address these gaps "
        f"and weaknesses. Focus on high-priority items first. Keep the total to 3-5 follow-ups."
    )

    output: FollowUpsOutput = structured.invoke([
        SystemMessage(content=GENERATE_FOLLOWUPS_SYSTEM),
        HumanMessage(content=prompt),
    ])

    by_priority: dict[str, list[str]] = {"high": [], "medium": [], "low": []}
    for f in output.followups:
        by_priority[f.priority].append(f.name)

    print(f"  → {len(output.followups)} follow-ups:")
    for p, names in by_priority.items():
        if names:
            print(f"     {p}: {', '.join(names)}")

    return {"followups_output": output}
