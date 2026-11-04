"""Agent-facing tool: erase an attribute with LEACE at every residual-stream site and re-measure.

The erasers are fitted on `fit_prompts` / `fit_counterfactual_prompts` and
tested on `prompts` / `counterfactual_prompts`, so they can't pass by being
fitted to the pairs they're scored on. Without fit prompts, every other pair
is used for fitting and the rest are held out. The erasers live only inside
this call; what comes back is how much the erasure changed.
"""

from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from seesaw.interp._helpers import tokens_to_ids
from seesaw.interp.metrics import last_positions, logit_diff
from seesaw.interp.model_editing import leace as _leace

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult


def _lds(model, prompts, pos_ids, neg_ids) -> torch.Tensor:
    with torch.no_grad():
        return logit_diff(model(model.to_tokens(prompts)), pos_ids, neg_ids,
                          last_positions(model, prompts)).cpu()


def _loss(model, texts) -> float:
    with torch.no_grad():
        return sum(model(model.to_tokens(t), return_type="loss").item() for t in texts) / len(texts)


def run_leace_erasure(
    model: HookedTransformer,
    prompts: list[str],
    counterfactual_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    fit_prompts: list[str] | None = None,
    fit_counterfactual_prompts: list[str] | None = None,
    control_texts: list[str] | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Erase what separates prompts from counterfactuals (LEACE, every site at once) and measure the reliance left.

    Reliance is the mean |logit diff change| from swapping the attribute;
    the task is the prompts' mean logit diff. Random erasers of the same rank
    are the control, with a loss check on control text (default: the prompts).
    Answers: "Can the attribute be removed without breaking the task?"
    """
    if fit_prompts is None or fit_counterfactual_prompts is None:
        if len(prompts) < 2:
            raise ValueError("leace_erasure needs at least 2 pairs to fit on some and test on the rest")
        fit, test = range(0, len(prompts), 2), range(1, len(prompts), 2)
        fit_prompts = [prompts[i] for i in fit]
        fit_counterfactual_prompts = [counterfactual_prompts[i] for i in fit]
        prompts = [prompts[i] for i in test]
        counterfactual_prompts = [counterfactual_prompts[i] for i in test]
        positive_tokens = [positive_tokens[i] for i in test]
        negative_tokens = [negative_tokens[i] for i in test]
    pos_ids, neg_ids = tokens_to_ids(model, positive_tokens), tokens_to_ids(model, negative_tokens)
    erasers = _leace.fit_all(model, fit_prompts, fit_counterfactual_prompts)
    texts = control_texts or prompts

    def measure(run_model):
        a, b = _lds(run_model, prompts, pos_ids, neg_ids), _lds(run_model, counterfactual_prompts, pos_ids, neg_ids)
        return (a - b).abs().mean().item(), a.mean().item(), _loss(run_model, texts)

    before = measure(model)
    with _leace.erased(model, erasers):
        after = measure(model)
    with _leace.erased(model, _leace.random_erasers(erasers)):
        rand = measure(model)

    def removed(m):
        return 1 - m[0] / before[0] if before[0] > 1e-6 else 0.0

    kept = after[1] / before[1] if abs(before[1]) > 1e-6 else 0.0
    return ExperimentResult(
        name="LEACE Erasure", tool="leace_erasure", model_name=model.cfg.model_name, prompts=prompts,
        data={
            "reliance_before": round(before[0], 4), "reliance_after": round(after[0], 4),
            "reliance_after_random": round(rand[0], 4),
            "task_before": round(before[1], 4), "task_after": round(after[1], 4),
            "control_loss_before": round(before[2], 4), "control_loss_after": round(after[2], 4),
            "fraction_removed": round(removed(after), 4),
            "random_fraction_removed": round(removed(rand), 4),
            "task_kept": round(kept, 4),
            "n_fit": len(fit_prompts), "n_test": len(prompts),
        },
        status="success",
    )
