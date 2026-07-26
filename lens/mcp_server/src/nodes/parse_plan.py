from ..models.schemas import ParsedPlanModel
from ..tools import TOOL_REGISTRY, token_defaults_from_specs, tool_kwargs_guide
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
        f"Only include experiments whose tool is one of: {list(TOOL_REGISTRY.keys())}.\n\n"
        f"Each tool takes `model` and `prompts` automatically. Anything else goes in\n"
        f"tool_kwargs, using exactly these argument names — an experiment whose\n"
        f"tool_kwargs are missing or misnamed cannot run:\n"
        f"{tool_kwargs_guide()}\n\n"
        f"Give one positive/negative token per prompt, in the same order as prompts\n"
        f"(a single pair is applied to every prompt). Token strings need their leading\n"
        f"space, e.g. ' Mary' not 'Mary'.\n"
        f"If no specific prompts are given, generate 2-3 appropriate IOI-style prompts.\n\n"
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
        "results":           [],
        "followup_count":    0,
        "last_result":       None,
        "bundle":            None,
    }
