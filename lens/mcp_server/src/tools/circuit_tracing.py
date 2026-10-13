"""Agent-facing wrapper over seesaw.interp.circuit_finding.circuit_tracing.

Attribution graphs come from Neuronpedia, which generates them for
gemma-2-2b only — whatever model the rest of the plan uses. The result says
so, so nothing reads a Gemma graph as a statement about GPT-2.

Generating a graph publishes the prompt on Neuronpedia, so it happens only
when SEESAW_NEURONPEDIA_PUBLISH=1; otherwise the tool reads an existing graph
(by `slug`, or the one an earlier run generated for this prompt) and fails
with an explanation if there is none.
"""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.circuit_finding import circuit_tracing as _circuit_tracing
from seesaw.interp.feature_extraction.neuronpedia import NeuronpediaError

from ..config import NEURONPEDIA_PUBLISH, PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap

GRAPH_MODEL = "gemma-2-2b"


def run_circuit_tracing(
    model: HookedTransformer,
    prompts: list[str],
    slug: str | None = None,
    top_k: int = 15,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Trace the attribution graph for the first prompt on Neuronpedia (gemma-2-2b) and summarise it.

    Reports the top transcoder features by influence on the predicted token,
    the strongest edges between them, and how much influence runs through
    error nodes the transcoders don't explain. An estimate, not an intervention.
    Answers: "Which features, through which intermediate steps, produce this token?"
    """
    try:
        result = _circuit_tracing.compute(
            output_dir, prompt=prompts[0], slug=slug, model_id=GRAPH_MODEL,
            allow_generate=NEURONPEDIA_PUBLISH, top_k=top_k)
    except NeuronpediaError as exc:
        if NEURONPEDIA_PUBLISH or slug:
            raise
        raise NeuronpediaError(
            f"no attribution graph for this prompt on Neuronpedia yet, and Lens doesn't generate one "
            f"unless SEESAW_NEURONPEDIA_PUBLISH=1 (generating publishes the prompt). Pass `slug` to "
            f"read an existing graph instead. ({exc})") from None
    result.data["traced_model"] = f"{GRAPH_MODEL} (on Neuronpedia)"
    if model.cfg.model_name != GRAPH_MODEL:
        result.data["note"] = (f"traced on {GRAPH_MODEL}, not {model.cfg.model_name}: Neuronpedia "
                               f"only generates graphs for {GRAPH_MODEL}")
    return wrap(result, "Circuit Tracing", "circuit_tracing", model, prompts[:1])
