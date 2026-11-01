"""Agent-facing wrapper over seesaw.interp.probing.linear_probe."""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.probing import linear_probe as _linear_probe

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def run_linear_probe(
    model: HookedTransformer,
    prompts: list[str],
    counterfactual_prompts: list[str],
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Train a probe at every layer to separate `prompts` from `counterfactual_prompts`.

    Decodable is not the same as used: follow with directional_ablation.
    Answers: "Where in the model is this concept represented?"
    """
    result = _linear_probe.compute(model, prompts, counterfactual_prompts, output_dir)
    return wrap(result, "Linear Probe", "linear_probe", model, prompts, drop=("direction",))
