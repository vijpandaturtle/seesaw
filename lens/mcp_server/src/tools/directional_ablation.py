"""Agent-facing wrapper over seesaw.interp.model_editing.directional_ablation.

The direction is mean(fit_prompts) - mean(fit_counterfactual_prompts) at
`layer` (the linear probe's best layer if not given), built here for the
same reason as in steering_vector; the fit prompts default to the prompts.
With `bake`, the removal is also folded into the weights and the gap
re-measured, to check the edit could ship as a checkpoint; the weights are
restored before the call returns.
"""

from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from seesaw.interp._helpers import tokens_to_ids
from seesaw.interp.metrics import last_positions, logit_diff
from seesaw.interp.model_editing import directional_ablation as _directional_ablation
from seesaw.interp.probing import linear_probe as _linear_probe

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def _gap(model, prompts, counterfactual_prompts, pos_ids, neg_ids) -> float:
    with torch.no_grad():
        a = logit_diff(model(model.to_tokens(prompts)), pos_ids, neg_ids,
                       last_positions(model, prompts)).mean().item()
        b = logit_diff(model(model.to_tokens(counterfactual_prompts)), pos_ids, neg_ids,
                       last_positions(model, counterfactual_prompts)).mean().item()
    return a - b


def run_directional_ablation(
    model: HookedTransformer,
    prompts: list[str],
    counterfactual_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    layer: int | None = None,
    fit_prompts: list[str] | None = None,
    fit_counterfactual_prompts: list[str] | None = None,
    control_texts: list[str] | None = None,
    bake: bool = False,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Remove the prompts-vs-counterfactuals direction everywhere and re-measure the gap.

    Compared against random directions, with a loss check on control text.
    Answers: "Does the model need this direction for the behaviour?"
    """
    fit_a = fit_prompts or prompts
    fit_b = fit_counterfactual_prompts or counterfactual_prompts
    if layer is None:
        layer = _linear_probe.compute(model, fit_a, fit_b, output_dir).data["best_layer"]
    direction = _linear_probe.direction(model, fit_a, fit_b, layer)
    result = _directional_ablation.compute(
        model, prompts, positive_tokens, negative_tokens, direction, output_dir,
        counterfactual_prompts=counterfactual_prompts, control_texts=control_texts)
    result.data["direction_layer"] = layer
    if bake:
        pos_ids, neg_ids = tokens_to_ids(model, positive_tokens), tokens_to_ids(model, negative_tokens)
        with _directional_ablation.orthogonalized(model, direction):
            baked = _gap(model, prompts, counterfactual_prompts, pos_ids, neg_ids)
        result.data["baked_effect_after"] = round(baked, 4)
    return wrap(result, f"Directional Ablation (layer {layer})", "directional_ablation", model, prompts)
