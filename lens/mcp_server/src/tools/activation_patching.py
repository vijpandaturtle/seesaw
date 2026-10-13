"""Agent-facing wrapper over seesaw.interp.causal_interventions.activation_patching.

The real computation lives in the library; this just adapts its InterpResult
into the ExperimentResult shape Lens's graph and Quill expect.
"""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.causal_interventions import activation_patching as _activation_patching

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult


def run_activation_patching(
    model: HookedTransformer,
    prompts: list[str],
    corrupted_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Patch the residual stream from the clean run into the corrupted run,
    layer by layer and position by position, and measure logit diff recovery.

    High recovery (→ 1.0) = that layer+position causally carries the IO information.
    Answers: "Which layer and token position carries the critical information?"

    Args:
        model: Loaded HookedTransformer.
        prompts: Clean prompts (model gets the right answer).
        corrupted_prompts: Corrupted prompts (model gets the wrong answer, e.g. names swapped).
        positive_tokens: Tokens to boost (e.g. [" Mary"] for IOI, [" she"] for gender bias).
        negative_tokens: Tokens to suppress (e.g. [" John"] for IOI, [" he"] for gender bias).
        output_dir: Directory to save the plot.

    Returns:
        ExperimentResult with layer × position recovery heatmap.
    """
    result = _activation_patching.compute(
        model, prompts, corrupted_prompts, positive_tokens, negative_tokens, output_dir,
    )
    return ExperimentResult(
        name="Activation Patching",
        tool="activation_patching",
        model_name=model.cfg.model_name,
        prompts=prompts,
        plot_paths=result.plot_paths,
        data=result.data,
        status="success",
    )
