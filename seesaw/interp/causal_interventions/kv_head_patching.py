"""KV-head patching: patch one key/value head's keys and values from the clean run into the corrupted run.

In grouped-query attention (Llama, Qwen, Gemma-2) a group of query heads
shares one key head and one value head. Head patching moves one query head's
output, but what the group reads is shared: if the information sits in the
keys or values, every head in the group gets it at once. This patches each
KV head's keys (hook_k), values (hook_v), and both, alone, and reports which
query heads read from it.

On a model without grouped-query attention every head is its own KV head,
so this is per-head key/value patching.

Answers: "Does the signal reach this group through what it attends to (keys)
or through what it moves (values)?"
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from .._helpers import tokens_to_ids
from ..metrics import logit_diff, recovery
from ..result import InterpResult


def n_kv_heads(model: HookedTransformer) -> int:
    return getattr(model.cfg, "n_key_value_heads", None) or model.cfg.n_heads


def group(model: HookedTransformer, layer: int, kv: int) -> list[str]:
    """The query heads ('L3H4', ...) that read KV head `kv` of `layer`."""
    size = model.cfg.n_heads // n_kv_heads(model)
    return [f"L{layer}H{h}" for h in range(kv * size, (kv + 1) * size)]


def compute(
    model: HookedTransformer,
    prompts: list[str],
    corrupted_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
    layers: list[int] | None = None,
) -> InterpResult:
    """Patch each KV head's keys, values and both, alone, from clean into corrupted.

    Args:
        model: Loaded HookedTransformer.
        prompts: Clean prompts.
        corrupted_prompts: Position-aligned corrupted prompts.
        positive_tokens: Correct answer token per prompt.
        negative_tokens: Distractor token per prompt.
        output_dir: Unused (no plot); kept so every tool has the same shape.
        layers: Layers to patch (default: all).

    Returns:
        InterpResult with each KV head ('L3KV1') mapped to its recovery from
        keys, values and both (0 = none, 1 = the whole gap) and the query heads
        that read it, plus the top KV heads by recovery from both.
    """
    pos_ids, neg_ids = tokens_to_ids(model, positive_tokens), tokens_to_ids(model, negative_tokens)
    clean_tokens, corrupt_tokens = model.to_tokens(prompts), model.to_tokens(corrupted_prompts)
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError("clean and corrupted prompts must tokenize to the same shape")
    layers = list(range(model.cfg.n_layers)) if layers is None else [int(l) for l in layers]
    names = {f"blocks.{l}.attn.hook_{x}" for l in layers for x in "kv"}
    with torch.no_grad():
        clean_logits, cache = model.run_with_cache(clean_tokens, names_filter=lambda n: n in names)
        clean_ld = logit_diff(clean_logits, pos_ids, neg_ids).mean().item()
        corrupt_ld = logit_diff(model(corrupt_tokens), pos_ids, neg_ids).mean().item()

    def patched(layer: int, kv: int, kinds: str) -> float:
        def put(value, hook):
            value = value.clone()
            value[:, :, kv] = cache[hook.name][:, :, kv]
            return value

        hooks = [(f"blocks.{layer}.attn.hook_{x}", put) for x in kinds]
        with torch.no_grad(), model.hooks(fwd_hooks=hooks):
            ld = logit_diff(model(corrupt_tokens), pos_ids, neg_ids).mean().item()
        return round(recovery(ld, corrupt_ld, clean_ld), 4)

    per_kv_head = {}
    for layer in layers:
        for kv in range(n_kv_heads(model)):
            per_kv_head[f"L{layer}KV{kv}"] = {
                "keys": patched(layer, kv, "k"), "values": patched(layer, kv, "v"),
                "both": patched(layer, kv, "kv"), "query_heads": group(model, layer, kv),
            }
    top = sorted(per_kv_head, key=lambda k: -per_kv_head[k]["both"])[:10]
    return InterpResult(data={
        "per_kv_head":   per_kv_head,
        "top_kv_heads":  [(k, per_kv_head[k]["both"]) for k in top],
        "grouped_query": n_kv_heads(model) != model.cfg.n_heads,
        "heads_per_group": model.cfg.n_heads // n_kv_heads(model),
        "clean_ld":      round(clean_ld, 4),
        "corrupt_ld":    round(corrupt_ld, 4),
    })
