from ..models.schemas import ParsedPlanModel
from ..tools import TOOL_REGISTRY
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
        f"Only include experiments whose tool is one of: {list(TOOL_REGISTRY.keys())}.\n"
        f"For tools that need io_tokens and subject_tokens, include them in tool_kwargs.\n"
        f"If no specific prompts are given, generate 2-3 appropriate IOI-style prompts.\n\n"
        f"Research Plan:\n{state['research_plan']}"
    )
    parsed = structured_llm.invoke(prompt)
    queue  = [e.model_dump() for e in parsed.experiments]
    print(f"   Found {len(queue)} experiments: {[e['name'] for e in queue]}")
    return {
        "research_question": parsed.research_question,
        "model_name":        parsed.model_name,
        "experiment_queue":  queue,
        "results":           [],
        "followup_count":    0,
        "last_result":       None,
        "bundle":            None,
    }
