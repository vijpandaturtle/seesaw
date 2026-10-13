"""Head patching: patch named heads from the clean run into the corrupted run and measure recovery.

The verification half of nominate-then-verify. Attribution patching (or DLA)
produces a shortlist; this runs real forward passes on just those heads —
each one alone, then all of them together — so the cost scales with the
shortlist, not the model.

The joint number matters as much as the individual ones: heads in a circuit
are often redundant (backup name movers in IOI), so each alone can recover
little while together they recover most of the gap.

Answers: "Do these specific heads actually carry the signal?"
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from .._helpers import parse_head, tokens_to_ids
from ..metrics import logit_diff, recovery
from ..result import InterpResult


def _patched_ld(model, corrupt_tokens, clean_z, heads, pos_ids, neg_ids) -> float:
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)

    def make_hook(layer_heads):
        def hook_fn(value, hook):
            value = value.clone()
            value[:, :, layer_heads, :] = clean_z[hook.name][:, :, layer_heads, :]
            return value
        return hook_fn

    hooks = [(f"blocks.{l}.attn.hook_z", make_hook(hs)) for l, hs in by_layer.items()]
    with torch.no_grad(), model.hooks(fwd_hooks=hooks):
        logits = model(corrupt_tokens)
    return logit_diff(logits, pos_ids, neg_ids).mean().item()


def compute(
    model: HookedTransformer,
    prompts: list[str],
    corrupted_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    heads: list[str],
    output_dir: Path,
) -> InterpResult:
    """Patch each listed head (and then all of them jointly) from clean into corrupted.

    Args:
        model: Loaded HookedTransformer.
        prompts: Clean prompts.
        corrupted_prompts: Position-aligned corrupted prompts.
        positive_tokens: Correct answer token per prompt.
        negative_tokens: Distractor token per prompt.
        heads: Heads to test, e.g. ["L9H9", "L9H6"].
        output_dir: Directory to save the plot.

    Returns:
        InterpResult with per-head and joint recovery (0 = none, 1 = full).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_tokens   = model.to_tokens(prompts)
    corrupt_tokens = model.to_tokens(corrupted_prompts)
    pos_ids = tokens_to_ids(model, positive_tokens)
    neg_ids = tokens_to_ids(model, negative_tokens)
    parsed = [parse_head(h) for h in heads]

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(
            clean_tokens, names_filter=lambda n: n.endswith("hook_z"))
        corrupt_logits = model(corrupt_tokens)
    clean_ld   = logit_diff(clean_logits, pos_ids, neg_ids).mean().item()
    corrupt_ld = logit_diff(corrupt_logits, pos_ids, neg_ids).mean().item()

    per_head = {}
    for name, lh in zip(heads, parsed):
        ld = _patched_ld(model, corrupt_tokens, clean_cache, [lh], pos_ids, neg_ids)
        per_head[name] = round(recovery(ld, corrupt_ld, clean_ld), 4)
    joint_ld = _patched_ld(model, corrupt_tokens, clean_cache, parsed, pos_ids, neg_ids) if parsed else corrupt_ld
    joint = round(recovery(joint_ld, corrupt_ld, clean_ld), 4)
    del clean_cache

    # ── Plot ──────────────────────────────────────────────────────────────────
    labels = list(per_head) + ["all together"]
    values = list(per_head.values()) + [joint]
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels) + 2), 4))
    colors = ["#4575b4"] * len(per_head) + ["#d73027"]
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Fraction of logit diff recovered")
    ax.set_title("Head Patching — clean → corrupted")
    plt.tight_layout()
    plot_path = output_dir / "head_patching.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "clean_ld":       round(clean_ld, 4),
            "corrupt_ld":     round(corrupt_ld, 4),
            "per_head":       per_head,
            "joint_recovery": joint,
            "heads":          heads,
            "corrupted_prompts": corrupted_prompts,
        },
    )
