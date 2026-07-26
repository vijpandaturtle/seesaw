"""Notebook-friendly wrappers over Lens's TransformerLens tools.

Each function takes a model *name* (loaded and cached on first use) instead of a
HookedTransformer instance, so a researcher can run one experiment in two lines:

    from seesaw import experiments
    r = experiments.ablation("gpt2",
                             prompts=["When Mary and John went to the store, John gave a drink to"],
                             positive_tokens=[" Mary"], negative_tokens=[" John"])
    r.data["top_heads"]      # [("L9H6", 0.9), ...]
    r.plot_paths             # saved heatmap PNGs

Requires the [experiments] extra (torch + transformer-lens). Returns the same
ExperimentResult objects the Lens agent produces, so anything graded by the eval
suite can also be produced interactively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from lens.mcp_server.src.models.schemas import ExperimentResult

__all__ = ["load_model", "logit_lens", "attention_pattern",
           "direct_logit_attribution", "ablation", "activation_patching"]


def load_model(model_name: str = "gpt2"):
    """Load (and cache) a TransformerLens model by name."""
    from lens.mcp_server.src.app.model_session import get_model

    return get_model(model_name)


def logit_lens(model_name: str, prompts: list[str], **kwargs) -> "ExperimentResult":
    from lens.mcp_server.src.tools import run_logit_lens

    return run_logit_lens(load_model(model_name), prompts, **kwargs)


def attention_pattern(model_name: str, prompts: list[str], **kwargs) -> "ExperimentResult":
    from lens.mcp_server.src.tools import run_attention_pattern

    return run_attention_pattern(load_model(model_name), prompts, **kwargs)


def direct_logit_attribution(model_name: str, prompts: list[str],
                             positive_tokens: list[str], negative_tokens: list[str],
                             **kwargs) -> "ExperimentResult":
    from lens.mcp_server.src.tools import run_direct_logit_attribution

    return run_direct_logit_attribution(load_model(model_name), prompts,
                                        positive_tokens, negative_tokens, **kwargs)


def ablation(model_name: str, prompts: list[str],
             positive_tokens: list[str], negative_tokens: list[str],
             **kwargs) -> "ExperimentResult":
    from lens.mcp_server.src.tools import run_ablation

    return run_ablation(load_model(model_name), prompts,
                        positive_tokens, negative_tokens, **kwargs)


def activation_patching(model_name: str, prompts: list[str], corrupted_prompts: list[str],
                        positive_tokens: list[str], negative_tokens: list[str],
                        **kwargs) -> "ExperimentResult":
    from lens.mcp_server.src.tools import run_activation_patching

    return run_activation_patching(load_model(model_name), prompts, corrupted_prompts,
                                   positive_tokens, negative_tokens, **kwargs)
