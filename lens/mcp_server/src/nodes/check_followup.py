from ..config import MAX_FOLLOWUPS
from ..models.schemas import FollowUpDecision
from ..tools import TOOL_REGISTRY
from ..utils import make_llm


def check_followup(state: dict) -> dict:
    """Decide whether the last result warrants an immediate follow-up experiment.

    Node 5 in the Lens workflow.
    """
    result         = state["last_result"]
    followup_count = state.get("followup_count", 0)
    if followup_count >= MAX_FOLLOWUPS or result["status"] != "success":
        return {}

    llm            = make_llm()
    structured_llm = llm.with_structured_output(FollowUpDecision)
    decision       = structured_llm.invoke(
        f"Based on this result, decide if an immediate follow-up is needed.\n"
        f"Be conservative — only if the result reveals something a different tool can directly test.\n"
        f"Available tools: {list(TOOL_REGISTRY.keys())}\n"
        f"Follow-ups used: {followup_count}/{MAX_FOLLOWUPS}\n"
        f"Summary: {result['summary']}"
    )

    if decision.needs_followup and decision.followup_tool:
        new_spec = {
            "name":             f"Follow-up: {decision.followup_description or decision.followup_tool}",
            "tool":             decision.followup_tool,
            "model_name":       result["model_name"],
            "prompts":          decision.followup_prompts or result["prompts"],
            "what_to_measure":  decision.followup_description or "",
            "hypothesis_tested": "follow-up",
            "expected_outcome": "",
            "tool_kwargs":      decision.followup_kwargs,
        }
        print(f"   ➕ Follow-up: {new_spec['name']}")
        return {
            "experiment_queue": [new_spec] + list(state["experiment_queue"]),
            "followup_count":   followup_count + 1,
        }
    return {}
