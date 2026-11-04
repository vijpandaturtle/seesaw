"""Agent-facing wrappers over the fact-locating tools: causal_tracing and rome_edits.

Both take factual prompts with their subjects written exactly as they appear
in the prompt ("The Eiffel Tower is located in the city of" / "The Eiffel
Tower"). rome_edits makes its edits one at a time and restores the weights
after each, so the edited model never outlives the call: what comes back is
how well each edit worked.
"""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.causal_interventions import causal_tracing as _causal_tracing
from seesaw.interp.model_editing import rome as _rome

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def run_causal_tracing(
    model: HookedTransformer,
    prompts: list[str],
    subjects: list[str],
    positive_tokens: list[str],
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Noise the subject's embeddings, then restore the residual stream, MLPs or attention at each layer and position.

    Recovery is the share of the answer's probability that comes back.
    Answers: "Where does the model retrieve this fact: which layers, at which token?"
    """
    result = _causal_tracing.compute(model, prompts, subjects, positive_tokens, output_dir)
    return wrap(result, "Causal Tracing", "causal_tracing", model, prompts)


def run_rome_edits(
    model: HookedTransformer,
    prompts: list[str],
    subjects: list[str],
    positive_tokens: list[str],
    target_tokens: list[str],
    paraphrase_prompts: list[str] | None = None,
    neighborhood_prompts: list[str] | None = None,
    neighborhood_tokens: list[str] | None = None,
    layers: list[int] | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Rewrite each fact to its target answer with ROME at each layer (default: all), and score the edits.

    Scored on efficacy (the prompt now gives the target), paraphrase (a
    rewording does too) and specificity (a neighbouring fact is untouched).
    Answers: "At which layer can this fact be edited, and cleanly?"
    """
    layers = list(range(model.cfg.n_layers)) if layers is None else [int(l) for l in layers]
    per_layer = {}
    for layer in layers:
        r = _rome.compute(model, prompts, subjects, positive_tokens, target_tokens, layer, output_dir,
                          paraphrase_prompts=paraphrase_prompts,
                          neighborhood_prompts=neighborhood_prompts,
                          neighborhood_tokens=neighborhood_tokens)
        per_layer[str(layer)] = {k: r.data.get(k) for k in ("efficacy", "paraphrase", "specificity", "score")}
    best = max(per_layer, key=lambda l: per_layer[l]["score"])
    return ExperimentResult(
        name="ROME Edits", tool="rome_edits", model_name=model.cfg.model_name, prompts=prompts,
        data={"per_layer": per_layer, "best_layer": int(best), "best_score": per_layer[best]["score"]},
        status="success",
    )
