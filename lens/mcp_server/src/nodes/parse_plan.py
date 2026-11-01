from shared.lens_tools import active_backend, bullet_list, planning_notes

from ..models.schemas import ParsedPlanModel
from ..tools import (
    prompt_pairs_from_specs, registry, token_defaults_from_specs, tool_kwargs_guide,
)

# What to do when the plan gives no inputs, per backend.
_DEFAULT_INPUTS = {
    "seesaw": "If no specific prompts are given, generate 2-3 appropriate IOI-style prompts.",
    "lever":  "If no specific images are given, build them from the synthetic forms "
              "('text:<words>', 'solid:<colour>', '+text:<words>') to fit the question.",
}
from ..utils import make_llm


def parse_plan(state: dict) -> dict:
    """Extract experiment specs from a Scout research plan.

    Node 1 in the Lens workflow.
    """
    print("📋 [parse_plan] Extracting experiments...")
    llm = make_llm()
    structured_llm = llm.with_structured_output(ParsedPlanModel)
    prompt = (
        f"Extract the experiments from this Research Plan.\n"
        f"For each experiment, extract the tool name, model, prompts, and what to measure.\n"
        f"Only include experiments whose tool is one of: {list(registry())}.\n"
        f"{planning_notes()}\n"
        f"What each tool does:\n{bullet_list()}\n\n"
        f"Each tool takes `model` and `prompts` automatically. Anything else goes in\n"
        f"tool_kwargs, using exactly these argument names — an experiment whose\n"
        f"tool_kwargs are missing or misnamed cannot run:\n"
        f"{tool_kwargs_guide()}\n\n"
        f"Give one positive/negative token per prompt, in the same order as prompts\n"
        f"(a single pair is applied to every prompt). Corrupted and counterfactual prompts\n"
        f"are one per prompt, in the same order, and must tokenize to the same length.\n"
        f"Token strings need exactly the spacing the model would produce: a leading space\n"
        f"mid-sentence in a text model (' Mary' not 'Mary'), none at the start of a chat answer.\n"
        f"{_DEFAULT_INPUTS[active_backend()]}\n\n"
        f"Research Plan:\n{state['research_plan']}"
    )
    parsed = structured_llm.invoke(prompt)
    queue  = [e.model_dump() for e in parsed.experiments]
    tokens = token_defaults_from_specs(queue)
    print(f"   Found {len(queue)} experiments: {[e['name'] for e in queue]}")
    if tokens:
        print(f"   Token pair: {tokens['positive_tokens']} vs {tokens['negative_tokens']}")
    return {
        "research_question": parsed.research_question,
        "model_name":        parsed.model_name,
        "experiment_queue":  queue,
        "token_defaults":    tokens,
        "prompt_pairs":      prompt_pairs_from_specs(queue),
        "results":           [],
        "followup_count":    0,
        "last_result":       None,
        "bundle":            None,
    }
