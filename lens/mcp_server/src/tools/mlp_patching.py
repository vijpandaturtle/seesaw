"""Agent-facing wrapper over seesaw.interp.causal_interventions.mlp_patching."""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.causal_interventions import mlp_patching as _mlp_patching

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def run_mlp_patching(
    model: HookedTransformer,
    prompts: list[str],
    corrupted_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    layers: list[int] | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Patch each MLP layer's output (default: all) from the clean run into the corrupted run, one at a time.

    Answers: "Which MLP layers carry the signal?"
    """
    result = _mlp_patching.compute(model, prompts, corrupted_prompts, positive_tokens, negative_tokens,
                                   output_dir, layers=None if layers is None else [int(l) for l in layers])
    return wrap(result, "MLP Patching", "mlp_patching", model, prompts)
