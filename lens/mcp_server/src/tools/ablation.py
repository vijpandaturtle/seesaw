"""Agent-facing wrapper over seesaw.interp.causal_interventions.ablation.

The real computation lives in the library; this just adapts its InterpResult
into the ExperimentResult shape Lens's graph and Quill expect.
"""

from pathlib import Path
from typing import Literal

from transformer_lens import HookedTransformer

from seesaw.interp.causal_interventions import ablation as _ablation

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult


def run_ablation(
    model: HookedTransformer,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    ablation_type: Literal["zero", "mean", "resample"] = "mean",
    heads: list[str] | None = None,
    corrupted_prompts: list[str] | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Ablate each attention head (or only `heads`) and measure the drop in logit diff.

    Large positive drop = causally important head.
    Answers: "Is this component causally necessary for the behaviour?"

    Args:
        model: Loaded HookedTransformer.
        prompts: List of input prompt strings.
        positive_tokens: Tokens to boost (e.g. [" Mary"] for IOI, [" she"] for gender bias).
        negative_tokens: Tokens to suppress (e.g. [" John"] for IOI, [" he"] for gender bias).
        ablation_type: "zero" to zero out the head, "mean" to replace with mean
            activation, "resample" to replace with its output on corrupted_prompts.
        heads: Only ablate these heads, e.g. ["L9H9", "L10H7"]; default every head.
        corrupted_prompts: Position-aligned prompts to resample from; needed for "resample".
        output_dir: Directory to save the plot.

    Returns:
        ExperimentResult with ablation heatmap and top important heads.
    """
    result = _ablation.compute(
        model, prompts, positive_tokens, negative_tokens, output_dir,
        ablation_type=ablation_type, heads=heads, resample_prompts=corrupted_prompts,
    )
    return ExperimentResult(
        name=f"Ablation ({ablation_type})",
        tool="ablation",
        model_name=model.cfg.model_name,
        prompts=prompts,
        plot_paths=result.plot_paths,
        data=result.data,
        status="success",
    )
