"""Agent-facing wrapper over seesaw.interp.visualization.attention_pattern.

The real computation lives in the library; this just adapts its InterpResult
into the ExperimentResult shape Lens's graph and Quill expect.
"""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.visualization import attention_pattern as _attention_pattern

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult


def run_attention_pattern(
    model: HookedTransformer,
    prompts: list[str],
    layers: list[int] | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Visualise attention weights per head per layer.

    Answers: "What does each head attend to?"

    Args:
        model: Loaded HookedTransformer.
        prompts: List of input prompt strings.
        layers: Specific layers to visualise. Defaults to 5 evenly-spaced layers.
        output_dir: Directory to save plots.

    Returns:
        ExperimentResult with one plot per layer.
    """
    result = _attention_pattern.compute(model, prompts, output_dir, layers=layers)
    return ExperimentResult(
        name="Attention Pattern",
        tool="attention_pattern",
        model_name=model.cfg.model_name,
        prompts=prompts,
        plot_paths=result.plot_paths,
        data=result.data,
        status="success",
    )
