"""Notebook-friendly wrappers over seesaw.interp — the mech interp toolkit itself.

Each function takes a model *name* (loaded and cached on first use) instead of a
HookedTransformer instance, so a researcher can run one experiment in two lines:

    from seesaw import experiments
    r = experiments.ablation("gpt2",
                             prompts=["When Mary and John went to the store, John gave a drink to"],
                             positive_tokens=[" Mary"], negative_tokens=[" John"])
    r.data["top_heads"]      # [("L9H6", 0.9), ...]
    r.plot_paths             # saved heatmap PNGs

Requires the [experiments] extra (torch + transformer-lens). Returns the same
InterpResult objects seesaw.interp itself produces — no agent framing (no
`tool`/`name`/`status`/`summary`), just `data` and `plot_paths`. This calls
straight into the library now rather than through Lens's agent-wrapped tools:
Lens adds a `summary` slot for its own LLM interpretation step and registers
these as MCP tools, neither of which a notebook session needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from seesaw.interp.result import InterpResult

__all__ = ["load_model", "logit_lens", "attention_pattern",
           "direct_logit_attribution", "ablation", "activation_patching"]


def _default_output_dir() -> "Path":
    # Reuse Lens's own PLOTS_DIR rather than guessing a path — it already
    # accounts for SEESAW_ARTIFACTS_DIR (Modal's mounted volume) vs a plain
    # local outputs/ folder. Imported lazily so plain `import seesaw` still
    # pulls in neither torch nor this settings module.
    from lens.mcp_server.src.config import PLOTS_DIR

    return PLOTS_DIR


def load_model(model_name: str = "gpt2"):
    """Load (and cache) a TransformerLens model by name."""
    from lens.mcp_server.src.app.model_session import get_model

    return get_model(model_name)


def logit_lens(model_name: str, prompts: list[str], **kwargs) -> "InterpResult":
    from seesaw.interp.visualization import logit_lens as _logit_lens

    kwargs.setdefault("output_dir", _default_output_dir())
    return _logit_lens.compute(load_model(model_name), prompts, **kwargs)


def attention_pattern(model_name: str, prompts: list[str], **kwargs) -> "InterpResult":
    from seesaw.interp.visualization import attention_pattern as _attention_pattern

    kwargs.setdefault("output_dir", _default_output_dir())
    return _attention_pattern.compute(load_model(model_name), prompts, **kwargs)


def direct_logit_attribution(model_name: str, prompts: list[str],
                             positive_tokens: list[str], negative_tokens: list[str],
                             **kwargs) -> "InterpResult":
    from seesaw.interp.visualization import direct_logit_attribution as _dla

    kwargs.setdefault("output_dir", _default_output_dir())
    return _dla.compute(load_model(model_name), prompts,
                        positive_tokens, negative_tokens, **kwargs)


def ablation(model_name: str, prompts: list[str],
             positive_tokens: list[str], negative_tokens: list[str],
             **kwargs) -> "InterpResult":
    from seesaw.interp.causal_interventions import ablation as _ablation

    kwargs.setdefault("output_dir", _default_output_dir())
    return _ablation.compute(load_model(model_name), prompts,
                             positive_tokens, negative_tokens, **kwargs)


def activation_patching(model_name: str, prompts: list[str], corrupted_prompts: list[str],
                        positive_tokens: list[str], negative_tokens: list[str],
                        **kwargs) -> "InterpResult":
    from seesaw.interp.causal_interventions import activation_patching as _activation_patching

    kwargs.setdefault("output_dir", _default_output_dir())
    return _activation_patching.compute(load_model(model_name), prompts, corrupted_prompts,
                                        positive_tokens, negative_tokens, **kwargs)
