"""Agent-facing wrapper over seesaw.interp.hidden_state_decoding.patchscope."""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.hidden_state_decoding import patchscope as _patchscope

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def run_patchscope(
    model: HookedTransformer,
    prompts: list[str],
    positive_tokens: list[str] | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Decode the last token's hidden state at every layer by patching it into an identity prompt.

    Reported beside the logit lens's reading of the same state, which
    under-reads early layers on bigger models.
    Answers: "From which layer does the hidden state already encode the answer?"
    """
    result = _patchscope.compute(model, prompts, output_dir, positive_tokens=positive_tokens)
    return wrap(result, "Patchscope", "patchscope", model, prompts)
