"""Agent-facing wrapper over seesaw.interp.visualization.logit_lens.

The real computation lives in the library; this just adapts its InterpResult
into the ExperimentResult shape Lens's graph and Quill expect.
"""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.visualization import logit_lens as _logit_lens

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult


def run_logit_lens(
    model: HookedTransformer,
    prompts: list[str],
    pos: int = -1,
    top_k: int = 5,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Project residual stream at each layer to vocab space.

    Shows how the model's prediction evolves layer by layer.
    Answers: "Where does the model 'decide' on the answer?"

    Args:
        model: Loaded HookedTransformer.
        prompts: List of input prompt strings.
        pos: Token position to inspect (-1 = last token).
        top_k: Number of top tokens to track per layer.
        output_dir: Directory to save the plot.

    Returns:
        ExperimentResult with a heatmap plot and layer-by-layer top-token data.
    """
    result = _logit_lens.compute(model, prompts, output_dir, pos=pos, top_k=top_k)
    return ExperimentResult(
        name="Logit Lens",
        tool="logit_lens",
        model_name=model.cfg.model_name,
        prompts=prompts,
        plot_paths=result.plot_paths,
        data=result.data,
        status="success",
    )
