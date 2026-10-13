"""Agent-facing wrapper over seesaw.interp.model_editing.directional_ablation.

The direction is mean(prompts) - mean(counterfactual_prompts) at `layer`
(the linear probe's best layer if not given), built here for the same reason
as in steering_vector.
"""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.model_editing import directional_ablation as _directional_ablation
from seesaw.interp.probing import linear_probe as _linear_probe

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def run_directional_ablation(
    model: HookedTransformer,
    prompts: list[str],
    counterfactual_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    layer: int | None = None,
    control_texts: list[str] | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Remove the prompts-vs-counterfactuals direction everywhere and re-measure the gap.

    Compared against random directions, with a loss check on control text.
    Answers: "Does the model need this direction for the behaviour?"
    """
    if layer is None:
        layer = _linear_probe.compute(model, prompts, counterfactual_prompts, output_dir).data["best_layer"]
    direction = _linear_probe.direction(model, prompts, counterfactual_prompts, layer)
    result = _directional_ablation.compute(
        model, prompts, positive_tokens, negative_tokens, direction, output_dir,
        counterfactual_prompts=counterfactual_prompts, control_texts=control_texts)
    result.data["direction_layer"] = layer
    return wrap(result, f"Directional Ablation (layer {layer})", "directional_ablation", model, prompts)
