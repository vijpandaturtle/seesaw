"""Agent-facing wrapper over seesaw.interp.steering.steering_vector.

An agent can't hand over a d_model-sized vector, so the direction is built
here from the two prompt sets: mean(prompts) - mean(counterfactual_prompts)
in the residual stream at `layer` (the linear probe's best layer if not
given). It's added to the counterfactual prompts, so alpha = 1 asks "does
adding the full difference make them behave like the prompts?"
"""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.probing import linear_probe as _linear_probe
from seesaw.interp.steering import steering_vector as _steering_vector

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def run_steering_vector(
    model: HookedTransformer,
    prompts: list[str],
    counterfactual_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    layer: int | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Add the prompts-minus-counterfactuals direction to the counterfactual prompts and sweep its strength.

    Answers: "Does pushing along this direction change the behaviour, and smoothly?"
    """
    if layer is None:
        layer = _linear_probe.compute(model, prompts, counterfactual_prompts, output_dir).data["best_layer"]
    direction = _linear_probe.direction(model, prompts, counterfactual_prompts, layer)
    n = len(counterfactual_prompts)
    result = _steering_vector.compute(
        model, counterfactual_prompts, positive_tokens[:n], negative_tokens[:n], direction, layer,
        output_dir, alphas=[-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0])
    return wrap(result, f"Steering Vector (layer {layer})", "steering_vector", model, prompts)
