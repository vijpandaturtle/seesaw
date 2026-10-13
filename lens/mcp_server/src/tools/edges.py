"""Agent-facing wrappers over the edge-level circuit tools: edge_attribution_patching and path_patching."""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.circuit_finding import edge_attribution_patching as _eap
from seesaw.interp.circuit_finding import path_patching as _path_patching

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def run_edge_attribution_patching(
    model: HookedTransformer,
    prompts: list[str],
    corrupted_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    top_k: int = 30,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Estimate every sender → receiver edge's patching effect (EAP-IG) and draw the top edges.

    An estimate: confirm the edges with path_patching.
    Answers: "Which connections between heads carry the behaviour?"
    """
    result = _eap.compute(model, prompts, corrupted_prompts, positive_tokens, negative_tokens,
                          output_dir, top_k=top_k)
    return wrap(result, "Edge Attribution Patching", "edge_attribution_patching", model, prompts)


def run_path_patching(
    model: HookedTransformer,
    prompts: list[str],
    corrupted_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    edges: list[str],
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Patch each listed edge alone (and all together) from clean into corrupted.

    Answers: "Does this specific connection carry the behaviour?"
    """
    result = _path_patching.compute(model, prompts, corrupted_prompts, positive_tokens,
                                    negative_tokens, edges, output_dir)
    return wrap(result, "Path Patching", "path_patching", model, prompts)
