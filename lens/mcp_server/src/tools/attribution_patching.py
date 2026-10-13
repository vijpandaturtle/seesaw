"""Agent-facing wrapper over seesaw.interp.causal_interventions.attribution_patching."""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.causal_interventions import attribution_patching as _atp

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def run_attribution_patching(
    model: HookedTransformer,
    prompts: list[str],
    corrupted_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    top_k: int = 10,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Estimate every head's and MLP's activation-patching effect from one backward pass.

    A linear estimate: use it to pick heads for head_patching, never as the
    causal effect itself.
    Answers: "Which heads are worth testing causally?"
    """
    result = _atp.compute(model, prompts, corrupted_prompts, positive_tokens, negative_tokens,
                          output_dir, top_k=top_k)
    return wrap(result, "Attribution Patching", "attribution_patching", model, prompts,
                drop=("head_effects",))
