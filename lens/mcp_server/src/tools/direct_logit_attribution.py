"""Agent-facing wrapper over seesaw.interp.visualization.direct_logit_attribution.

The real computation lives in the library; this just adapts its InterpResult
into the ExperimentResult shape Lens's graph and Quill expect.
"""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.visualization import direct_logit_attribution as _dla

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult


def run_direct_logit_attribution(
    model: HookedTransformer,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    pos: int = -1,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Decompose the final logit difference into per-head and per-MLP contributions.

    Answers: "Which components *write* the IO prediction into the residual stream?"

    Args:
        model: Loaded HookedTransformer.
        prompts: List of input prompt strings.
        positive_tokens: Tokens to boost (e.g. [" Mary"] for IOI, [" she"] for gender bias).
        negative_tokens: Tokens to suppress (e.g. [" John"] for IOI, [" he"] for gender bias).
        pos: Token position to read logits from (-1 = last token).
        output_dir: Directory to save the plot.

    Returns:
        ExperimentResult with head and MLP attribution heatmaps.
    """
    result = _dla.compute(
        model, prompts, positive_tokens, negative_tokens, output_dir, pos=pos
    )
    return ExperimentResult(
        name="Direct Logit Attribution",
        tool="direct_logit_attribution",
        model_name=model.cfg.model_name,
        prompts=prompts,
        plot_paths=result.plot_paths,
        data=result.data,
        status="success",
    )
