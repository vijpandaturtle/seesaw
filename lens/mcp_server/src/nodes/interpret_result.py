import json

from shared.evidence import as_text, assess, how_to_report

from ..utils import make_llm


def data_preview(data: dict, limit: int = 2_000) -> str:
    """The data as JSON, short values first, so a large matrix can't push the
    headline numbers (top_heads, baselines) past the cut-off."""
    ordered = dict(sorted(data.items(), key=lambda kv: len(json.dumps(kv[1], default=str))))
    return json.dumps(ordered, indent=2, default=str)[:limit]


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
    preview      = data_preview(result["data"])
    evidence     = result.get("evidence") or assess(result["tool"], result["data"], len(result["prompts"]))
    response     = llm.invoke(
        f"You are an AI safety researcher specialised in mechanistic interpretability.\n"
        f"Interpret this experiment result in 2-3 paragraphs:\n"
        f"1. What do the results reveal about the model's internal mechanisms?\n"
        f"2. What is the most important finding?\n"
        f"3. How does this connect to the research question: \"{state['research_question']}\"\n\n"
        f"Open with one sentence, in plain words a non-expert can follow, saying how strong this "
        f"evidence is and why. If a required control is missing, say the result is incomplete, name "
        f"what's missing, and don't draw the conclusion it would have supported. Respect the caveats.\n\n"
        f"Experiment: {result['name']} | Tool: {result['tool']}\n"
        f"{how_to_report(result['tool'])}\n"
        f"{as_text(evidence)}\n"
        f"Data: {preview}"
    )
    result["summary"] = response.content
    return {"results": list(state["results"]) + [result], "last_result": result}
