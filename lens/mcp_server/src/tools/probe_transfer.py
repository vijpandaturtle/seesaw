"""Agent-facing wrapper over seesaw.interp.probing.transfer.

The second class is `counterfactual_prompts`, as in linear_probe, so a plan's
prompt pairs route here the same way.
"""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.probing import transfer as _transfer

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def run_probe_transfer(
    model: HookedTransformer,
    prompts: list[str],
    counterfactual_prompts: list[str],
    test_prompts: list[str],
    test_counterfactual_prompts: list[str],
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Train a probe on prompts vs counterfactual_prompts, choose its layer there, and score it on held-out texts.

    Compared with shuffled labels, a probe on the embeddings alone and a
    bag-of-words classifier with no model.
    Answers: "Would a probe on the activations make a good detector, better than the words alone?"
    """
    result = _transfer.compute(model, prompts, counterfactual_prompts, test_prompts,
                               test_counterfactual_prompts, output_dir)
    return wrap(result, "Probe Transfer", "probe_transfer", model, prompts)
