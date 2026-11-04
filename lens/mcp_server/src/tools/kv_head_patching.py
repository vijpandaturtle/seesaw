"""Agent-facing wrapper over seesaw.interp.causal_interventions.kv_head_patching."""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.causal_interventions import kv_head_patching as _kv_head_patching

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def run_kv_head_patching(
    model: HookedTransformer,
    prompts: list[str],
    corrupted_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    layers: list[int] | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Patch each key/value head's keys, values and both from clean into corrupted, one at a time.

    In grouped-query models (Llama, Qwen, Gemma) a KV head serves a group of
    query heads; the result names the group ('L3KV1': L3H4..L3H7).
    Answers: "Does the signal reach these heads through their keys or their values?"
    """
    result = _kv_head_patching.compute(model, prompts, corrupted_prompts, positive_tokens,
                                       negative_tokens, output_dir, layers=layers)
    return wrap(result, "KV-Head Patching", "kv_head_patching", model, prompts)
