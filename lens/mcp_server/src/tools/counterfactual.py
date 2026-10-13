"""Agent-facing wrapper over seesaw.interp.black_box.counterfactual."""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.black_box import counterfactual as _counterfactual

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def run_counterfactual(
    model: HookedTransformer,
    prompts: list[str],
    counterfactual_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    control_prompts: list[str] | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Swap one attribute in each prompt and measure how far the logit diff moves.

    Answers: "Does this attribute change the model's output, and how consistently?"
    """
    result = _counterfactual.compute(model, prompts, counterfactual_prompts, positive_tokens,
                                     negative_tokens, output_dir, control_prompts=control_prompts)
    return wrap(result, "Counterfactual Evaluation", "counterfactual", model, prompts)
