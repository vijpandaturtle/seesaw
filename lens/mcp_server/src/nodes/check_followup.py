from ..config import MAX_FOLLOWUPS
from ..models.schemas import FollowUpDecision
from ..tools import KWARG_ALIASES, TOOL_REGISTRY, tool_kwargs_guide
from ..utils import make_llm

# Question-level arguments a follow-up inherits when it doesn't name its own.
_INHERITED = ("positive_tokens", "negative_tokens")


def check_followup(state: dict) -> dict:
    """Decide whether the last result warrants an immediate follow-up experiment.

    Node 5 in the Lens workflow.

    The follow-up is dispatched by the same code path as a planned experiment,
    so it has to satisfy the same tool contract — hence the kwargs guide in the
    prompt. Token pairs describe the behaviour under study rather than one
    experiment, so they carry over from the plan unless the follow-up names
    its own.
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
        f"Available tools: {list(TOOL_REGISTRY.keys())}\n\n"
        f"followup_kwargs must use exactly these argument names — a follow-up with\n"
        f"missing or misnamed kwargs cannot run:\n"
        f"{tool_kwargs_guide()}\n"
        f"These tools sweep every head and layer; there is no way to restrict them to\n"
        f"specific heads, so don't ask for that — read the sweep's heatmap instead.\n"
        f"Reuse the same positive/negative tokens as the experiment you're following up on.\n\n"
        f"Follow-ups used: {followup_count}/{MAX_FOLLOWUPS}\n"
        f"Summary: {result['summary']}"
    )

    if decision.needs_followup and decision.followup_tool:
        kwargs = {
            KWARG_ALIASES.get(k, k): v
            for k, v in (decision.followup_kwargs or {}).items()
        }
        inherited = state.get("token_defaults") or {}
        for arg in _INHERITED:
            if arg not in kwargs and arg in inherited:
                kwargs[arg] = inherited[arg]

        new_spec = {
            "name":             f"Follow-up: {decision.followup_description or decision.followup_tool}",
            "tool":             decision.followup_tool,
            "model_name":       result["model_name"],
            "prompts":          decision.followup_prompts or result["prompts"],
            "what_to_measure":  decision.followup_description or "",
            "hypothesis_tested": "follow-up",
            "expected_outcome": "",
            "tool_kwargs":      kwargs,
        }
        print(f"   ➕ Follow-up: {new_spec['name']}")
        return {
            "experiment_queue": [new_spec] + list(state["experiment_queue"]),
            "followup_count":   followup_count + 1,
        }
    return {}
