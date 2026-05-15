from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..app.helpers import bundle_to_context
from ..app.schemas import ExperimentCritiquesOutput
from ..config import (
    ANTHROPIC_API_KEY,
    CRITIQUE_EXPERIMENTS_SYSTEM,
    QUILL_MODEL,
    QUILL_TEMPERATURE,
)


def critique_experiments(state: dict) -> dict:
    """Assess each experiment for validity, confounds, and whether conclusions are supported.

    Node 1 in the Quill workflow.
    """
    bundle = state["bundle"]
    print("\n🔬 [critique_experiments] Assessing each experiment...")

    model = ChatAnthropic(
        model=QUILL_MODEL,
        temperature=QUILL_TEMPERATURE,
        api_key=ANTHROPIC_API_KEY,
    )
    structured = model.with_structured_output(ExperimentCritiquesOutput)

    ctx    = bundle_to_context(bundle)
    prompt = (
        f"Please critique each of the following experiments.\n\n"
        f"{ctx}\n\n"
        f"Assess every successful experiment individually.\n"
        f"Then give an overall assessment of the experiment set as a whole."
    )

    output: ExperimentCritiquesOutput = structured.invoke([
        SystemMessage(content=CRITIQUE_EXPERIMENTS_SYSTEM),
        HumanMessage(content=prompt),
    ])

    print(f"  → Overall: {output.overall_assessment}")
    for c in output.critiques:
        supported = "✓" if c.conclusions_supported else "✗"
        print(
            f"  → {c.experiment_name}: {c.validity} | "
            f"conclusions_supported={supported} | issues={len(c.issues)}"
        )

    return {"critiques_output": output}
