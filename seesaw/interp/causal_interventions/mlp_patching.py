"""MLP patching: patch each MLP layer's output from the clean run into the corrupted run.

The MLP counterpart of head patching, for verifying the MLP layers that
attribution patching nominates. Each layer is patched alone, at every
position, and the logit diff's recovery is recorded.

Answers: "Do the MLP layers carry part of the signal, and which ones?"
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from .._helpers import tokens_to_ids
from ..metrics import logit_diff, recovery
from ..result import InterpResult


def compute(
    model: HookedTransformer,
    prompts: list[str],
    corrupted_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
    layers: list[int] | None = None,
) -> InterpResult:
    """Patch each listed MLP layer (default: all of them) alone, from clean into corrupted.

    Args:
        model: Loaded HookedTransformer.
        prompts: Clean prompts.
        corrupted_prompts: Position-aligned corrupted prompts.
        positive_tokens: Correct answer token per prompt.
        negative_tokens: Distractor token per prompt.
        output_dir: Unused (no plot); kept so every tool has the same shape.
        layers: MLP layers to patch.

    Returns:
        InterpResult with each layer's recovery (0 = none, 1 = the whole gap).
    """
    pos_ids = tokens_to_ids(model, positive_tokens)
    neg_ids = tokens_to_ids(model, negative_tokens)
    clean_tokens, corrupt_tokens = model.to_tokens(prompts), model.to_tokens(corrupted_prompts)
    if model.cfg.attn_only:
        return InterpResult(data={"per_layer": {}, "clean_ld": None, "corrupt_ld": None})
    layers = list(range(model.cfg.n_layers)) if layers is None else layers
    with torch.no_grad():
        clean_logits, cache = model.run_with_cache(
            clean_tokens, names_filter=lambda n: n.endswith("hook_mlp_out"))
        clean_ld = logit_diff(clean_logits, pos_ids, neg_ids).mean().item()
        corrupt_ld = logit_diff(model(corrupt_tokens), pos_ids, neg_ids).mean().item()

    per_layer = {}
    for layer in layers:
        name = f"blocks.{layer}.hook_mlp_out"
        with torch.no_grad(), model.hooks(fwd_hooks=[(name, lambda v, hook, n=name: cache[n])]):
            ld = logit_diff(model(corrupt_tokens), pos_ids, neg_ids).mean().item()
        per_layer[str(layer)] = round(recovery(ld, corrupt_ld, clean_ld), 4)
    return InterpResult(data={
        "per_layer":  per_layer,
        "clean_ld":   round(clean_ld, 4),
        "corrupt_ld": round(corrupt_ld, 4),
    })
