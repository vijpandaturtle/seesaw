from ..config import MAX_FOLLOWUPS
from ..tools import prompt_pairs_from_specs


def replay_plan(state: dict) -> dict:
    """Start from experiments already run, instead of reading them out of a plan.

    Node 1 in the Lens workflow when it is given `replay` (a rerun): the
    experiments a job ran before, each with its tool, prompts and arguments
    exactly as they ran. No LLM decides what to run, so the same experiments
    run again. Follow-ups are switched off, since the source job's follow-ups
    are already in the list; only the LLM-written summaries can differ.
    """
    replay = state["replay"]
    queue  = [dict(spec) for spec in replay["experiments"]]
    print(f"📋 [replay_plan] Replaying {len(queue)} experiment(s) as they ran before")
    return {
        "research_question": replay["research_question"],
        "model_name":        replay["model_name"],
        "experiment_queue":  queue,
        "token_defaults":    {},
        "prompt_pairs":      prompt_pairs_from_specs(queue),
        "results":           [],
        "followup_count":    MAX_FOLLOWUPS,
        "last_result":       None,
        "bundle":            None,
    }
