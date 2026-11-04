"""Agent-facing wrapper over seesaw.interp.model_diffing.diff.

The changed model is named, not passed: a model can't be sent to the GPU app,
so it's loaded here the way Lens loads every model (cached after the first
load). In-process callers may pass a loaded HookedTransformer instead.
"""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.model_diffing import diff as _diff

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def run_model_diff(
    model: HookedTransformer,
    prompts: list[str],
    other_model: str | HookedTransformer,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Compare the model with `other_model` (a fine-tune, quantisation or edit of it) on the prompts.

    KL between their next-token predictions, the prompts and tokens that
    changed most, and (same architecture) how much of the KL swapping each
    component's weights into the base closes.
    Answers: "What did the change do, and which weights did it?"
    """
    if isinstance(other_model, str):
        from ..app.model_session import get_model

        other_model = get_model(other_model)
    result = _diff.compute(model, other_model, prompts, output_dir)
    result.data["other_model"] = other_model.cfg.model_name
    return wrap(result, "Model Diff", "model_diff", model, prompts)
