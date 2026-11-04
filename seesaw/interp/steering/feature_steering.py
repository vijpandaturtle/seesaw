"""SAE feature steering: push the residual stream along one SAE feature's decoder direction.

The constructive counterpart of feature clamping. Clamping asks "does the
output need this feature?"; steering asks "is this feature enough to move
it?" — Templeton et al. 2024's Golden Gate test, with a logit-diff readout.

The push is in units of the residual stream's own size: alpha = 1 adds a
vector as long as the average residual at the SAE's hook point, along the
feature's unit decoder direction. The sweep, the random-direction control
and the loss check are steering_vector's, so the numbers read the same way.

Answers: "Does turning this feature up change the behaviour, and more than a
random direction would?"
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from ..feature_extraction.sae import hook_name
from ..result import InterpResult
from . import steering_vector


def _resid_post_layer(model: HookedTransformer, hook: str) -> int:
    """The layer whose hook_resid_post is `hook` (resid_pre of layer L is resid_post of L - 1)."""
    parts = hook.split(".")
    layer, site = int(parts[1]), parts[-1]
    if site == "hook_resid_post":
        return layer
    if site == "hook_resid_pre" and layer > 0:
        return layer - 1
    raise ValueError(f"feature steering needs a residual-stream SAE after the embeddings, not {hook!r}")


def compute(
    model: HookedTransformer,
    sae,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    feature: int,
    output_dir: Path,
    alphas: list[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
    control_texts: list[str] | None = None,
) -> InterpResult:
    """Add `feature`'s decoder direction to the residual stream at the SAE's hook point, at each strength.

    Args:
        model: Loaded HookedTransformer.
        sae: A residual-stream SAE (see feature_extraction.sae.load_sae).
        prompts: Prompts to steer.
        positive_tokens: Token per prompt the feature should push toward.
        negative_tokens: Token per prompt it should push away from.
        feature: Index of the feature to steer with.
        output_dir: Directory to save the plot.
        alphas: Strengths, in multiples of the mean residual norm at the hook.
        control_texts: Text to measure loss on. Defaults to `prompts`.

    Returns:
        steering_vector's InterpResult (logit diff per alpha, slope, random
        directions' slope, loss), plus the feature, hook and push size.
    """
    hook = hook_name(sae)
    layer = _resid_post_layer(model, hook)
    with torch.no_grad():
        _, cache = model.run_with_cache(model.to_tokens(prompts), names_filter=lambda n: n == hook)
        scale = cache[hook].float().norm(dim=-1).mean().item()
    unit = sae.W_dec[int(feature)].detach().float()
    unit = unit / unit.norm()
    result = steering_vector.compute(model, prompts, positive_tokens, negative_tokens,
                                     (unit * scale).tolist(), layer, output_dir,
                                     alphas=list(alphas), control_texts=control_texts)
    result.data.update({"feature": int(feature), "hook_name": hook, "resid_norm": round(scale, 4)})
    return result
