"""Agent-facing wrapper over seesaw.interp.causal_interventions.head_patching."""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.causal_interventions import head_patching as _head_patching

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def run_head_patching(
    model: HookedTransformer,
    prompts: list[str],
    corrupted_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    heads: list[str],
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Patch the named heads from the clean run into the corrupted run, each alone and all together.

    Answers: "Do these specific heads actually carry the signal?"
    """
    result = _head_patching.compute(model, prompts, corrupted_prompts, positive_tokens,
                                    negative_tokens, heads, output_dir)
    return wrap(result, "Head Patching", "head_patching", model, prompts)
