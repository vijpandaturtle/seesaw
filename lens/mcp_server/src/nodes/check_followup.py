from shared.lens_tools import bullet_list, tool_chains

from ..config import MAX_FOLLOWUPS
from ..models.schemas import FollowUpDecision
from ..tools import KWARG_ALIASES, tool_kwargs_guide
from ..utils import make_llm
from .interpret_result import data_preview

# Question-level arguments a follow-up inherits when it doesn't name its own.
_INHERITED = ("positive_tokens", "negative_tokens")
# Images go with the prompts they were paired with (lever backend); the tools
# echo them in their data, so a follow-up on the same prompts reuses them.
_IMAGE_ARGS = ("images", "corrupted_images")


def check_followup(state: dict) -> dict:
    """Decide whether the last result warrants an immediate follow-up experiment.

    Node 5 in the Lens workflow.

    The follow-up is dispatched by the same code path as a planned experiment,
    so it has to satisfy the same tool contract — hence the kwargs guide in the
    prompt. Token pairs describe the behaviour under study rather than one
    experiment, so they carry over from the plan unless the follow-up names
    its own. Prompts always carry over: the token pairs are aligned to them
    per prompt, and left to choose, the LLM fills prompts with research
    questions ("What does L9H9 attend to?") that then run as model input.
    Their corrupted/counterfactual partners carry over with them, from the
    plan, since the LLM writing the follow-up never saw them.
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
        f"Available tools:\n{bullet_list()}\n\n"
        f"A natural follow-up checks an estimate or a descriptive result with a causal tool:\n"
        f"{tool_chains()}\n\n"
        f"followup_kwargs must use exactly these argument names — a follow-up with\n"
        f"missing or misnamed kwargs cannot run:\n"
        f"{tool_kwargs_guide()}\n"
        f"To test specific heads, pass `heads` named as they appear in the summary (e.g. 'L9H9').\n"
        f"Reuse the same positive/negative tokens as the experiment you're following up on.\n\n"
        f"Follow-ups used: {followup_count}/{MAX_FOLLOWUPS}\n"
        f"Summary: {result['summary']}\n\n"
        f"Key numbers (exact head names and feature indices are here): {data_preview(result['data'], 1_200)}"
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
        for arg in _IMAGE_ARGS:
            if arg not in kwargs and (result.get("data") or {}).get(arg):
                kwargs[arg] = result["data"][arg]
        pair = (state.get("prompt_pairs") or {}).get("\n".join(result["prompts"]))
        if pair and not any(a in kwargs for a in ("corrupted_prompts", "counterfactual_prompts")):
            # normalize_tool_kwargs renames this to counterfactual_prompts for tools that use that name.
            kwargs["corrupted_prompts"] = pair

        new_spec = {
            "name":             f"Follow-up: {decision.followup_description or decision.followup_tool}",
            "tool":             decision.followup_tool,
            "model_name":       result["model_name"],
            "prompts":          result["prompts"],
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
