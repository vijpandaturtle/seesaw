"""Agent-facing wrapper over seesaw.interp.steering.steering_vector.

An agent can't hand over a d_model-sized vector, so the direction is built
here from prompt sets: mean(fit_prompts) - mean(fit_counterfactual_prompts)
in the residual stream at `layer` (the linear probe's best layer if not
given). The fit prompts default to the prompts themselves; give separate
ones to test a direction on pairs it wasn't fitted to. It's added to the
counterfactual prompts, so alpha = 1 asks "does adding the full difference
make them behave like the prompts?"
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
    fit_prompts: list[str] | None = None,
    fit_counterfactual_prompts: list[str] | None = None,
    alphas: list[float] | None = None,
    control_texts: list[str] | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Add the prompts-minus-counterfactuals direction to the counterfactual prompts and sweep its strength.

    Answers: "Does pushing along this direction change the behaviour, and smoothly?"
    """
    fit_a = fit_prompts or prompts
    fit_b = fit_counterfactual_prompts or counterfactual_prompts
    if layer is None:
        # Only the layer is wanted here, so the probe's shuffled-label control is skipped.
        layer = _linear_probe.compute(model, fit_a, fit_b, output_dir, n_shuffles=0).data["best_layer"]
    direction = _linear_probe.direction(model, fit_a, fit_b, layer)
    n = len(counterfactual_prompts)
    result = _steering_vector.compute(
        model, counterfactual_prompts, positive_tokens[:n], negative_tokens[:n], direction, layer,
        output_dir, alphas=alphas or [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0], control_texts=control_texts)
    return wrap(result, f"Steering Vector (layer {layer})", "steering_vector", model, prompts)
