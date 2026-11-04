"""Agent-facing wrapper over seesaw.interp.circuit_finding.circuit_tracing.

Attribution graphs come from Neuronpedia, which generates them for a few
models (GRAPH_MODELS). A plan on one of them is traced on that model; any
other is traced on gemma-2-2b (or `graph_model`), and the result says so, so
nothing reads a Gemma graph as a statement about Llama.

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

# TransformerLens model name → Neuronpedia model id, for the models
# Neuronpedia generates attribution graphs for (docs.neuronpedia.org,
# circuit-tracing/generate-graphs). Llama-3.2 and Qwen2.5 aren't among them.
GRAPH_MODELS = {
    "gemma-2-2b": "gemma-2-2b",
    "gemma-3-4b-it": "gemma-3-4b-it",
    "Qwen3-4B": "qwen3-4b",
    "Qwen3-1.7B": "qwen3-1.7b",
}
GRAPH_MODEL = "gemma-2-2b"          # used when the plan's model has no graphs


def run_circuit_tracing(
    model: HookedTransformer,
    prompts: list[str],
    slug: str | None = None,
    top_k: int = 15,
    graph_model: str | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Trace the attribution graph for the first prompt on Neuronpedia and summarise it.

    Traced on the plan's model when Neuronpedia has graphs for it (gemma-2-2b,
    gemma-3-4b-it, Qwen3-4B, Qwen3-1.7B), else on `graph_model` (one of
    those Neuronpedia ids; default gemma-2-2b).

    Reports the top transcoder features by influence on the predicted token,
    the strongest edges between them, and how much influence runs through
    error nodes the transcoders don't explain. An estimate, not an intervention.
    Answers: "Which features, through which intermediate steps, produce this token?"
    """
    own = GRAPH_MODELS.get(model.cfg.model_name)
    traced = graph_model or own or GRAPH_MODEL
    if traced not in GRAPH_MODELS.values():
        raise ValueError(f"Neuronpedia generates graphs for {sorted(GRAPH_MODELS.values())}, not {traced!r}")
    try:
        result = _circuit_tracing.compute(
            output_dir, prompt=prompts[0], slug=slug, model_id=traced,
            allow_generate=NEURONPEDIA_PUBLISH, top_k=top_k)
    except NeuronpediaError as exc:
        if NEURONPEDIA_PUBLISH or slug:
            raise
        raise NeuronpediaError(
            f"no attribution graph for this prompt on Neuronpedia yet, and Lens doesn't generate one "
            f"unless SEESAW_NEURONPEDIA_PUBLISH=1 (generating publishes the prompt). Pass `slug` to "
            f"read an existing graph instead. ({exc})") from None
    result.data["traced_model"] = f"{traced} (on Neuronpedia)"
    if traced != own:
        result.data["note"] = (f"traced on {traced}, not {model.cfg.model_name}: Neuronpedia generates "
                               f"graphs only for {sorted(GRAPH_MODELS.values())}")
    return wrap(result, "Circuit Tracing", "circuit_tracing", model, prompts[:1])
