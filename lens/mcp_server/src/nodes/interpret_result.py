import json

from ..utils import make_llm


def interpret_result(state: dict) -> dict:
    """Ask an LLM to interpret the last experiment result in plain language.

    Node 4 in the Lens workflow.
    """
    result = state["last_result"]
    print(f"💬 [interpret_result] '{result['name']}'")

    if result["status"] != "success":
        result["summary"] = f"Experiment failed: {result.get('error')}"
        return {"results": list(state["results"]) + [result], "last_result": result}

    llm          = make_llm()
    data_preview = json.dumps(result["data"], indent=2)[:2_000]
    response     = llm.invoke(
        f"You are an AI safety researcher specialised in mechanistic interpretability.\n"
        f"Interpret this experiment result in 2-3 paragraphs:\n"
        f"1. What do the results reveal about the model's internal mechanisms?\n"
        f"2. What is the most important finding?\n"
        f"3. How does this connect to the research question: \"{state['research_question']}\"\n\n"
        f"Experiment: {result['name']} | Tool: {result['tool']}\n"
        f"Data: {data_preview}"
    )
    result["summary"] = response.content
    return {"results": list(state["results"]) + [result], "last_result": result}
